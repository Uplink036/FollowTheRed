IP=$1
ssh -t jetbot@$IP rm -rf ~/jetbot/notebooks/client/; echo Hello World!