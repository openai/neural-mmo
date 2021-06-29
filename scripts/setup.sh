read -p "DEPENDENCIES: this script requires Anaconda Python 3.8, gcc, zlib1g-dev, make, cmake, and build-essential. Installation will fail otherwise. Proceed [y/n]?" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
   [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
fi

#Install python packages
echo "Installing conda pip packages"
pip install -r scripts/requirements.txt

if [[ $1 == "--CORE_ONLY" ]]; then 
   echo "You have chosen not to install RLlib and associated dependencies for learned models"
elif [[ $1 == "" ]]; then
   echo "Installing additional RLlib dependencies..."
   pip install -r scripts/rllib_requirements.txt
   pip install ray[rllib]
   echo "Installing cuda torch"
   conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
else
   echo "Specify either --SERVER_ONLY or no argument"
   exit 1
fi

echo "Done. Errors? Check that dependencies have been met"
