var myPythonScriptPath = '../Python/pycv-socket.py'
var PythonShell = require('python-shell');
var pyshell = new PythonShell(myPythonScriptPath);
var app = require('express')();
var http = require('http').Server(app);
var io = require('socket.io')(http);

io.set('heartbeat interval', 2000);
io.set('heartbeat timeout', 5000);

// end the input stream and allow the process to exit
pyshell.end(function (err) {
    if (err){
        throw err;
    };

    console.log('finished');
});

app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});

io.on('connection', function(socket){
  console.log('a user connected');
  pyshell.on('message', function(message) {
  	//console.log(message);
  	io.sockets.emit('broadcast', {
  		payload:message
  	});
  });
});

io.on('disconnect', function(usr){
	console.log("User has disconnected ")
	console.log(usr)
});

http.listen(3000, function(){
  console.log('listening on *:3000');
});
