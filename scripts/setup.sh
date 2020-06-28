echo "Neural MMO setup; assumes Anaconda Python 3.7 and gcc"
conda install pip

if [[ $1 == "--SERVER_ONLY" ]]; then 
   echo "You have chosen not to install the graphical rendering client"
elif [[ $1 == "" ]]; then
   echo "Installing Neural MMO Client (Unity3D)..."
   git clone https://github.com/jsuarez5341/neural-mmo-client
   mv neural-mmo-client forge/embyr
else
   echo "Specify either --SERVER_ONLY or no argument"
   exit 1
fi

#Install python packages
pip install -r scripts/requirements.txt
pip install -U scripts/ray-0.9.0.dev0-cp37-cp37m-manylinux1_x86_64.whl
