var toml = require('toml');
var fs = require('fs');

module.exports = function (file) {
    var rawData = fs.readFileSync(file);

    // Should throw exception if the data is bad
    toml.parse(rawData.toString());
    console.log("Valid TOML file");
}