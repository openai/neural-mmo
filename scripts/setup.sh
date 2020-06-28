echo "Running Neural MMO setup"
echo "This may take a few minutes..."
echo "If you are on a VM, install basic utilities first (curl, python, git)"

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

echo "Installing Poetry to manage dependencies..."
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
source ~/.poetry/env
poetry config virtualenvs.create false

echo "Installing dependencies with Poetry..."
poetry install
