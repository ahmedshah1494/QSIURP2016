#!/usr/bin/expect

spawn ssh Ahmed@10.33.48.38
expect "Password:"
send "14September.\r"
expect "$"
send "echo 'Cloning repo'\r"
expect "$"
send "cd /Users/Ahmed/Downloads/QSIURP2016\r"
expect "$"
send "git pull\r"
expect "$"
interact
