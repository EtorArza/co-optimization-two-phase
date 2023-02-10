# https://aweirdimagination.net/2020/06/28/kill-child-jobs-on-script-exit/
import sys
import os
os.chdir("other_repos/tholiao")
sys.path.append(sys.path[0]+"/../other_repos/tholiao")
print(sys.path)

from main import cli_main


if sys.argv[1] == "--local_launch":
    cli_main(3)
