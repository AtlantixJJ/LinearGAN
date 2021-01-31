'use strict';

function Graph(document, name) {
  var canvas = document.getElementById(name);
  var ctx = canvas.getContext('2d');

  var width = canvas.width;
  var height = canvas.height;
  var offsetX = canvas.offsetLeft + parseInt(canvas.style.borderBottomWidth);
  var offsetY = canvas.offsetTop + parseInt(canvas.style.borderBottomWidth);

  ctx.fillStyle = '#000';
  ctx.lineCap = 'round';
  ctx.lineWidth = 5;

  var mouseDown = false;
  var prev = null;
  var has_image = false;

  this.setSize = function (height, width) {
    prev = ctx.lineWidth;
    canvas.width = width;
    canvas.height = height;
    ctx.lineCap = 'round';
    ctx.lineWidth = prev;
  }

  this.setHasImage = function (flag) {
    has_image = flag;
  }

  this.getImageData = function () {
    return canvas.toDataURL('image/png');
  };

  this.setCurrentColor = function (colorString) {
    ctx.strokeStyle = colorString;
  };

  this.setLineWidth = function (width) {
    ctx.lineWidth = width;
  };

  function drawLine(x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  };
  this.drawLine = drawLine;

  this.clear = function () {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  function getMouse(event) {
    var rect = event.target.getBoundingClientRect();
    var mouse = {
      x: (event.touches && event.touches[0] || event).clientX - rect.left,
      y: (event.touches && event.touches[0] || event).clientY - rect.top
    };
    if (mouse.x > rect.width || mouse.x < 0 || mouse.y > rect.height || mouse.y < 0)
      return null;
    return mouse;
  }
  
  function onMouseUp(event) {
    mouseDown = false;
  }

  function onMouseDown(event) {
    if (event.button == 2 || !has_image) {
      mouseDown = false;
      return;
    }
    if (mouseDown) { // double down, basically impossible
      onMouseUp(event);
      return;
    }
    prev = getMouse(event);
    if (prev != null) mouseDown = true;
  }

  function onMouseMove(event) {
    event.preventDefault();
    if (mouseDown && has_image) {
      var mouse = getMouse(event);
      if (mouse == null) {
        mouseDown = false;
        return;
      }
      drawLine(prev.x, prev.y, mouse.x, mouse.y);
      prev = mouse;
    }
  }

  this.onMouseDown = onMouseDown;
  this.onMouseMove = onMouseMove;
  this.onMouseUp = onMouseUp;

  canvas.addEventListener('mousedown',  onMouseDown, false);
  canvas.addEventListener('mouseup',    onMouseUp, false);
  canvas.addEventListener('mousemove',  onMouseMove, false);
  canvas.addEventListener('mouseout',   onMouseUp, false);
  canvas.addEventListener('touchstart', onMouseDown, false);
  canvas.addEventListener('touchend',   onMouseUp, false);
  canvas.addEventListener('touchmove',  onMouseMove, false);
  canvas.addEventListener('touchcancel',onMouseUp, false);
}