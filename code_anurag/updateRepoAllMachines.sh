#!/usr/bin/expect

set n 4
set pword 14September.
set timeout 120
# puts "$pword\r"
for {set i 2} {$i <= $n} {incr i 1} {

	set hname ""
	append hname "mshah1@nsl-n0" ${i} ".qatar.cmu.local"
	puts "trying $hname"
	set prompt "~\$ "
	puts "$prompt"
	spawn ssh $hname
	expect "?assword:"
	send "14September.\r"
	expect "\$ "
	send "scp mshah1@nsl-n01.qatar.cmu.local:~/QSIURP2016/ .\r"
	sleep 2
	expect "?assword:"
	send "14September.\r"
	sleep 1
	expect "\$ "
	send "echo 'exiting'\r"
	sleep 2
	expect "\$ "
	send "exit\r"
	sleep 3
}