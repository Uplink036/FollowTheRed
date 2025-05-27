IP=$1
ssh -t jetbot@$IP   pip install -r req.txt;
                    sudo apt update;
                    sudo apt install ffmpeg libsm6 libxext6 -y