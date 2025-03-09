# cs121_A3SearchEngine

## Set up
### Download data set 
- Follow link to download zip file containing data set of ics.uci.edu pages:
https://www.ics.uci.edu/~algol/teaching/informatics141cs121w2022/a3files/developer.zip

- Once downloaded extract files and place 'developer' folder into root project director

### Initialize Virtual Environment 
Creates lightweight python environment to manage project dependecies

Create virtual environment (bash)\
$ ```python -m venv .venv```

Activate virtual environment (bash)\
$ ```source .venv/bin/activate```

Activate virtual environment (windows)\
$ ```source .venv/Scripts/activate```

Deactivate virtual environment (bash)\
$ ```deactivate```

### Install Dependencies 
```python -m pip install -r packages/requirements.txt```

### Setting up VSCode environment
  Create ".vscode" folder. Add two json files.
  - launch.json - Configures debugging settings
  - settings.json - Configures unittesting setttings
  
  launch.json
``` 
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debug: main.py",
            "type": "debugpy",
            "request": "launch",
            "program": "ma${extensionInstallFolder:publisher.extension}.py",
            "console": "integratedTerminal",
            "args": []
        },
        {
            "name": "Python Debug: build_index.py",
            "type": "debugpy",
            "request": "launch",
            "program": "build_index.py",
            "console": "integratedTerminal",
            "args": [
                "--rootdir", "corpus"
            ]
        },
        {
            "name": "Python Debug: summary.py",
            "type": "debugpy",
            "request": "launch",
            "program": "summary.py",
            "console": "integratedTerminal",
            "args": []
        }
    ]
}
```

settings.json
```
{
    "python.testing.unittestArgs": [
        "-v",
        "-s",
        "./test",
        "-p",
        "*_test.py"
    ],
    "python.testing.pytestEnabled": false,
    "python.testing.unittestEnabled": true
}
```

### Verify the Environment
$ ```which python```

### Setup
$ ```python setup.py```

Downloads required nltk resources. Run setup.py once before building index or entering search query

### Building Index
$ ```python build_index.py --rootdir <file_path>```

Builds the inverted index. If the index is not already built run build_index.py. Script only needs to

optional arguments:
--rootdir <file_path> : File path to document corpus directory

### Running Search Engine and Entering a Query
$ ```python -m flask run```

Visit site
http://127.0.0.1:5000


