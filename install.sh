set -e

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update

sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo "🐍 Creating Python 3.10 virtual environment..."
python3.10 -m venv .venv
source .venv/bin/activate

echo "📦 Installing your project..."
pip install --upgrade pip
python setup.py install

echo "source .venv/bin/activate"
