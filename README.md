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
            "name": "Python Debug: main.py with restart arg",
            "type": "debugpy",
            "request": "launch",
            "program": "main.py",
            "console": "integratedTerminal",
            "args": [
                "--rootdir", "developer/DEV", "--restart"
            ]
        },
        {
            "name": "Python Debug: summary_report.py",
            "type": "debugpy",
            "request": "launch",
            "program": "summary_report.py",
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


## M1: Inverse Index


