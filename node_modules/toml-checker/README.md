# toml-checker
Simple library to validate TOML files

## Installation
To install the package globally:
```
npm install -g toml-checker
```

To install as a dev dependency:
```
npm install --save-dev toml-checker
```

## Usage
#### Run from command line:

If you installed as a global package: `toml-checker <path-to-toml-file>`

If you installed locally: `node node_modules/toml-checker/bin/toml-checker.js <path-to-toml-file>
`
#### Run from package.json

1. Add a script in your package.json that calls `toml-checker`
```
"scripts": {
    "test": "NODE_ENV=test mocha --timeout 5000 test/*.js",
    "check-toml": "toml-checker <path-to-toml-file>"
  }
  ```
2. Run the script: `npm run check-toml`
