#!/usr/bin/env node
'use strict';

var checkToml = require('../index');

var file = process.argv.length > 1 ? process.argv[2] : undefined;

checkToml(file);