import argparse
import ast
import csv
import datetime
import env
import mcts
import numpy as np
import os
import pyrobotdesign as rd
import random
import signal
import sys
import tasks

def get_applicable_matches(rule, graph):
  """Generates all applicable matches for rule in graph."""
  for match in rd.find_matches(rule.lhs, graph):
    if rd.check_rule_applicability(rule, graph, match):
      yield match

def has_nonterminals(graph):
  """Returns True if the graph contains nonterminal nodes/edges, and False
  otherwise."""
  for node in graph.nodes:
    if node.attrs.shape == rd.LinkShape.NONE:
      return True
  for edge in graph.edges:
    if edge.attrs.joint_type == rd.JointType.NONE:
      return True
  return False

def make_graph(rules, rule_sequence):
  graph = make_initial_graph()
  for r in rule_sequence:
    matches = list(get_applicable_matches(rules[r], graph))
    if matches:
      graph = rd.apply_rule(rules[r], graph, matches[0])
    else:
      raise ValueError("Rule in sequence has no applicable matches")
  return graph

def build_normalized_robot(graph):
  """Build a robot from the graph and normalize the mass of the body links."""
  robot = rd.build_robot(graph)

  body_links = []
  total_body_length = 0.0
  for link in robot.links:
    if np.isclose(link.radius, 0.045):
      # Link is a body link
      body_links.append(link)
      total_body_length += link.length
      target_mass = link.length * link.density

  if body_links:
    body_density = target_mass / total_body_length
    for link in body_links:
      link.density = body_density

  return robot

def presimulate(robot):
  """Find an initial position that will place the robot on the ground behind the
  x=0 plane, and check if the robot collides in its initial configuration."""
  temp_sim = rd.BulletSimulation()
  temp_sim.add_robot(robot, np.zeros(3), rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
  temp_sim.step()
  robot_idx = temp_sim.find_robot_index(robot)
  lower = np.zeros(3)
  upper = np.zeros(3)
  temp_sim.get_robot_world_aabb(robot_idx, lower, upper)
  return [-upper[0], -lower[1], 0.0], temp_sim.robot_has_collision(robot_idx)

def simulate(robot, task, opt_seed, thread_count, episode_count=1, no=None, test=False):
  """Run trajectory optimization for the robot on the given task, and return the
  resulting input sequence and result."""

  if no is None:
    raise ValueError("NestedOptimization object should not be None.")

  default_episode_len = no.params.default_inner_length
  default_nsamples = no.params.default_inner_length

  task.episode_len = default_episode_len if test else no.params.get_inner_length_absolute()
  nsamples = default_nsamples if test else no.params.get_inner_quantity_absolute()

  robot_init_pos, has_self_collision = presimulate(robot)

  if has_self_collision:
    return None, None

  def make_sim_fn():
    sim = rd.BulletSimulation(task.time_step)
    task.add_terrain(sim)
    # Rotate 180 degrees around the y axis, so the base points to the right
    sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    return sim

  main_sim = make_sim_fn()
  robot_idx = main_sim.find_robot_index(robot)

  dof_count = main_sim.get_robot_dof_count(robot_idx)
  links_count = main_sim.get_n_links(robot_idx)

  # dof stands for degrees of freedom (I think), and measures the amount of joints that can move
  # links_count counts the number of 'limbs' of the robot
  # print("dof_count", dof_count, "links_count",  links_count)

  controller_size = dof_count
  morphology_size =links_count



  value_estimator = rd.NullValueEstimator()
  input_sampler = rd.DefaultInputSampler()
  objective_fn = task.get_objective_fn()

  replay_obs = np.zeros((value_estimator.get_observation_size(), 0))
  replay_returns = np.zeros(0)
  for episode_idx in range(episode_count):
    optimizer = rd.MPPIOptimizer(
      1.0, 
      task.discount_factor, 
      dof_count,
      task.interval, 
      task.horizon, 
      512,
      thread_count, 
      opt_seed + episode_idx,
      make_sim_fn, 
      objective_fn, 
      value_estimator,
      input_sampler)

    optimizer.update()
    optimizer.set_sample_count(nsamples)

    main_sim.save_state()

    ObservedScores = np.zeros((nsamples, task.horizon), dtype=np.float64, order='F')
    input_sequence = np.zeros((dof_count, task.episode_len))
    obs = np.zeros((value_estimator.get_observation_size(),
                    task.episode_len + 1), order='f')
    rewards = np.zeros(task.episode_len * task.interval)
    for j in range(task.episode_len):

      optimizer.update()
      input_sequence[:,j] = optimizer.input_sequence[:,0]
      optimizer.advance(1)

      if obs.shape[0] > 0:
          value_estimator.get_observation(main_sim, obs[:,j])
      for k in range(task.interval):
        main_sim.set_joint_targets(robot_idx,
                                   input_sequence[:,j].reshape(-1, 1))
        task.add_noise(main_sim, j * task.interval + k)
        main_sim.step()
        rewards[j * task.interval + k] = objective_fn(main_sim)
    if obs.shape[0] > 0:
        value_estimator.get_observation(main_sim, obs[:,-1])

    main_sim.restore_state()

    for _ in range(task.episode_len):
      for _ in range(nsamples):
        no.next_step()
      no.next_inner()


    # Only train the value estimator if there will be another episode
    if episode_idx < episode_count - 1:
      returns = np.zeros(task.episode_len + 1)
      # Bootstrap returns with value estimator
      value_estimator.estimate_value(obs[:,task.episode_len], returns[-1:])
      for j in reversed(range(task.episode_len)):
        interval_reward = np.sum(
            rewards[j * task.interval:(j + 1) * task.interval])
        returns[j] = interval_reward + task.discount_factor * returns[j + 1]
      replay_obs = np.hstack((replay_obs, obs[:,:task.episode_len]))
      replay_returns = np.concatenate((replay_returns,
                                       returns[:task.episode_len]))
      value_estimator.train(replay_obs, replay_returns)
  if not test:
    no.next_outer(np.mean(rewards), controller_size, -1, morphology_size)
    if no.is_reevaluating: # If new best solution found...
      print("Reevaluating...")
      _, _ = simulate(robot, task, opt_seed, thread_count, episode_count=1, no=no, test=True)
    

  if test: # test=True means that no.next_outer() found a new best solution, and simulate was called recursively.
    from viewer import pickle_simulation_objects_for_video_generation

    reeval_f = np.mean(rewards)
    no.savenext_current=True
    no.next_reeval(reeval_f, controller_size, -1, morphology_size)
    pickle_simulation_objects_for_video_generation(
      opt_seed,
      task.taskname,
      input_sequence,
      dump_path=f"simulation_objects_{no.params.experiment_index}_current.pkl",
      visualization_path="../../results/robogrammar/videos/"+f"{no.get_video_label()}_current"+".mp4"
    )
    if no.save_best_visualization_required:
      no.savenext_best=True
      pickle_simulation_objects_for_video_generation(
        opt_seed,
        task.taskname,
        input_sequence,
        dump_path=f"simulation_objects_{no.params.experiment_index}_best.pkl",
      visualization_path="../../results/robogrammar/videos/"+f"{no.get_video_label()}_best"+".mp4"

      )
      no.save_best_visualization_required = False



    return None, None
      

  
  return input_sequence, np.mean(rewards)

def make_initial_graph():
  """Make an initial robot graph."""
  n0 = rd.Node()
  n0.name = 'robot'
  n0.attrs.label = 'robot'
  initial_graph = rd.Graph()
  initial_graph.nodes = [n0]
  return initial_graph

class RobotDesignEnv(env.Env):
  """Robot design environment where states are (graph, rule sequence) pairs and
  actions are rule applications."""

  def __init__(self, task, rules, seed, thread_count, max_rule_seq_len):
    self.task = task
    self.rules = rules
    self.rng = random.Random(seed)
    self.thread_count = thread_count
    self.max_rule_seq_len = max_rule_seq_len
    self.initial_graph = make_initial_graph()
    self.result_cache = dict()
    self.result_cache_hit_count = 0

  @property
  def initial_state(self):
    return (self.initial_graph, [])

  def get_available_actions(self, state):
    graph, rule_seq = state
    if len(rule_seq) >= self.max_rule_seq_len:
      # No more actions should be available
      return
    for rule in self.rules:
      if list(get_applicable_matches(rule, graph)):
        # Rule has at least one applicable match
        yield rule

  def get_next_state(self, state, rule):
    graph, rule_seq = state
    applicable_matches = list(get_applicable_matches(rule, graph))
    return (rd.apply_rule(rule, graph, applicable_matches[0]),
            rule_seq + [rule])

  def get_result(self, state, no):
    graph, rule_seq = state
    if has_nonterminals(graph):
      # Graph is incomplete
      return None
    robot = build_normalized_robot(graph)
    opt_seed = self.rng.getrandbits(32)
    self.latest_opt_seed = opt_seed
    result_cache_key = (tuple(self.rules.index(rule) for rule in rule_seq),
                        opt_seed)
    if result_cache_key in self.result_cache:
      result = self.result_cache[result_cache_key]
      self.result_cache_hit_count += 1
    else:
      _, result = simulate(robot, self.task, opt_seed, self.thread_count, 1, no)

    # FIXME: workaround for simulation instability
    # Simulation is invalid if the result is greater than result_bound
    if result is not None and result > self.task.result_bound:
      return None

    return result

  def get_key(self, state):
    return hash(state[0])

class RandomSearch(object):
  def __init__(self, env, max_tries):
    self.env = env
    self.max_tries = max_tries

  def select_action(self, state):
    available_actions = list(self.env.get_available_actions(state))
    if available_actions:
      return random.choice(available_actions)
    else:
      return None

  def run_iteration(self):
    result = None

    for try_count in range(self.max_tries):
      states = [self.env.initial_state]
      actions = []
      action = self.select_action(states[-1])
      while action is not None:
        states.append(self.env.get_next_state(states[-1], action))
        actions.append(action)
        action = self.select_action(states[-1])
      result = self.env.get_result(states[-1])
      if result is not None:
        # Result is valid
        break

    return states, actions, result

algorithms = {"mcts": mcts.TreeSearch, "random": RandomSearch}

def set_pdb_trace(sig, frame):
  import pdb
  pdb.Pdb().set_trace(frame)

def main(no, algorithm, cpus):
  signal.signal(signal.SIGUSR1, set_pdb_trace)

  args_list = ["--seed", str(no.params.seed),
               "-a", algorithm,
                f"-j{cpus}", # parallel threads
                f"-i1000000",
                "-d40",
                "--log_dir", "logs_mcts",
                f"{no.params.env_name}", "data/designs/grammar_apr30.dot",
              ]




  parser = argparse.ArgumentParser(description="Robot design search demo.")
  parser.add_argument("task", type=str, help="Task (Python class name)")
  parser.add_argument("grammar_file", type=str, help="Grammar file (.dot)")
  parser.add_argument("-a", "--algorithm", choices=algorithms.keys(),
                      default="mcts",
                      help="Algorithm ({})".format("|".join(algorithms.keys())))
  parser.add_argument("-s", "--seed", type=int, default=None,
                      help="Random seed")
  parser.add_argument("-j", "--jobs", type=int, required=True,
                      help="Number of jobs/threads")
  parser.add_argument("-i", "--iterations", type=int, required=True,
                      help="Number of iterations")
  parser.add_argument("-d", "--depth", type=int, required=True,
                      help="Maximum tree depth")
  parser.add_argument("-l", "--log_dir", type=str, default='',
                      help="Log directory")
  parser.add_argument("-f", "--log_file", type=str,
                      help="Existing log file, for resuming a previous run")
  args = parser.parse_args(args_list)

  random.seed(args.seed)
  taskname=no.params.env_name
  task_class = getattr(tasks, taskname)
  task = task_class()
  task.taskname=taskname
  graphs = rd.load_graphs(args.grammar_file)
  rules = [rd.create_rule_from_graph(g) for g in graphs]
  env = RobotDesignEnv(task, rules, args.seed, args.jobs, args.depth)
  search_alg = algorithms[args.algorithm](env, max_tries=1000)
  no.savenext_current=False
  no.savenext_best=False
  from viewer import pickle_rule_sequence_for_video_generation
  for i in range(args.iterations):
    states, actions, result = search_alg.run_iteration(no)
    if no.savenext_current:
      dump_path = f"rule_sequence_{no.params.experiment_index}_current.pkl"
      rule_seq = [rules.index(rule) for rule in actions]
      pickle_rule_sequence_for_video_generation(rule_seq, dump_path)
      no.savenext_current=False
    if no.savenext_best:
      dump_path = f"rule_sequence_{no.params.experiment_index}_best.pkl"
      rule_seq = [rules.index(rule) for rule in actions]
      pickle_rule_sequence_for_video_generation(rule_seq, dump_path)
      no.savenext_best=False
      
