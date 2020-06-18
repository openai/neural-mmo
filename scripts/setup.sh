echo "Running Neural MMO setup"
echo "Ensure you have gcc installed and go grab a coffee"
echo "This may take a few minutes..."

if $1 == "--SERVER_ONLY"
   echo "You have chosen not to install the graphical rendering client"
else
   echo "Installing Neural MMO Client (Unity3D)..."
   git clone https://github.com/jsuarez5341/neural-mmo-client
   mv neural-mmo-client forge/embyr
fi

echo "Installing Poetry to manage dependencies..."
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
source ~/.poetry/env

echo "Installing dependencies with Poetry..."
poetry install

echo "Generating game maps. This may take a few minutes..."
python -c "import terrain"
