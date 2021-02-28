'use strict';

function readTextFile(file, callback) {
  var rawFile = new XMLHttpRequest();
  rawFile.overrideMimeType("application/json");
  rawFile.open("GET", file, true);
  rawFile.onreadystatechange = function() {
      if (rawFile.readyState === 4 && rawFile.status == "200") {
          callback(rawFile.responseText);
      }
  }
  rawFile.send(null);
}

var annotator = null; // canvas manager
var loading = false; // the image is loading (waiting response)
var image = null; // image data
var config = null; // config file
var imwidth = 256, // image size
    imheight = 256; // image size
var use_args = false; // [deprecated]
var spinner = new Spinner({ color: '#999' });
var allModel = [];
var curModelID = null;
var category = [];

var COLORS = [
  'rgb(0, 0, 0)', 'rgb(255, 255, 0)', 'rgb(28, 230, 255)',
  'rgb(255, 52, 255)', 'rgb(255, 74, 70)', 'rgb(0, 137, 65)',
  'rgb(0, 111, 166)', 'rgb(163, 0, 89)', 'rgb(255, 219, 229)',
  'rgb(122, 73, 0)', 'rgb(0, 0, 166)', 'rgb(99, 255, 172)',
  'rgb(183, 151, 98)', 'rgb(0, 77, 67)', 'rgb(143, 176, 255)'];
var MAX_LINE_WIDTH = 20;


Date.prototype.format = function (format) {
  var o = {
    'M+': this.getMonth() + 1, //month
    'd+': this.getDate(), //day
    'h+': this.getHours(), //hour
    'm+': this.getMinutes(), //minute
    's+': this.getSeconds(), //second
    'q+': Math.floor((this.getMonth() + 3) / 3), //quarter
    S: this.getMilliseconds() //millisecond
  };
  if (/(y+)/.test(format)) format = format.replace(RegExp.$1, (this.getFullYear() + '').substr(4 - RegExp.$1.length));
  for (var k in o) {
    if (new RegExp('(' + k + ')').test(format)) format = format.replace(RegExp.$1, RegExp.$1.length == 1 ? o[k] : ('00' + o[k]).substr(('' + o[k]).length));
  }return format;
};

function setCategory(color) {
  annotator.setCurrentColor(color);
  $('#category-drop-menu .color-block').css('background-color', color);
  $('#category-drop-menu .color-block').css('border', color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none');
}

function setLineWidth(width) {
  annotator.setLineWidth(width * 2);
  $('#width-label').text(width);
}

function setModel(model) {
  if (loading) return;
  curModelID = model;
  $('#model-label').text(allModel[model]);
  onStart();
}

function setImage(data) {
  setLoading(false);
  if (!data || !data.ok) return;

  if (!image) {
    $('#stroke').removeClass('disabled');
    $('#option-buttons').prop('hidden', false);
  }
  image = data.img;
  $('#image').attr('src', image);
  $('#canvas').css('background-image', 'url(' + image + ')');
  annotator.setHasImage(true);
  spinner.spin();
  canvas_auto_resize();
}

function setLoading(isLoading) {
  loading = isLoading;
  annotator.setHasImage(false);
  $('#start-new').prop('disabled', loading);
  $('#submit').prop('disabled', loading);
  $('#spin').prop('hidden', !loading);
  if (loading) {
    $('.image').css('opacity', 0.7);
    spinner.spin(document.getElementById('spin'));
  } else {
    $('.image').css('opacity', 1);
    spinner.stop();
  }
}

function onSubmit() {
  if (annotator && !loading) {
    setLoading(true);
    var formData = {
      model: allModel[curModelID],
      annotation: annotator.getImageData(),
    };
    $.post('stroke', formData, setImage, 'json');
  }
}

function onStartNew() {
  if (loading) return;
  setLoading(true);
  annotator.clear();
  $.post('train/new',
    { model: allModel[curModelID] },
    setImage, 'json');
  console.log({ model: allModel[curModelID] });
}

function onStart() {
  onStartNew();
  $('#start').prop('hidden', true);
}

function setLabel(color) {
  annotator.setCurrentColor(color);
  $('#category-drop-menu .color-block').css('background-color', color);
  $('#category-drop-menu .color-block').css('border', color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none');
}

function init() {
  $('#color-menu').append(
    '\n<li role="presentation">\n  <div onclick="setColor(\'' +
    COLORS[0] +
    '\')"\n  >\n    <div class="color-block" style="background-color:' +
    COLORS[0] + ';border:' +
    (COLORS[0] == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none') +
    '"/>\n  </div>\n</li>');

  allModel.forEach(function (model, idx) {
    $('#model-menu').append(
      '<li role="presentation">\n' +
      '  <div class="dropdown-item model-item" onclick="setModel(' + idx + ')">' +
      model +
      '  </div>\n' +
      '</li>\n'
    );
  });

  var slider = document.getElementById('slider');
  noUiSlider.create(slider, {
    start: MAX_LINE_WIDTH / 2,
    step: 1,
    range: {
      'min': 1,
      'max': MAX_LINE_WIDTH
    },
    behaviour: 'drag-tap',
    connect: [true, false],
    orientation: 'vertical',
    direction: 'rtl',
  });
  slider.noUiSlider.on('update', function () {
    setLineWidth(parseInt(slider.noUiSlider.get()));
  });

  setLabel('black');

  setLineWidth(MAX_LINE_WIDTH / 2);

  $('#download-sketch').click(function () {
    download(
      annotator.getImageData(),
      'sketch_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#download-image').click(function () {
    download(
      image,
      'image_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#clear').click(annotator.clear);
  $('#submit').click(onSubmit);
  $('#stroke').click(function() {
    var stroke = $('#stroke').hasClass('active');
    if (stroke) {
      $('#image').prop('hidden', false);
      $('#canvas').prop('hidden', true);
      $('#stroke').removeClass('active');
      $('#stroke .btn-text').text('Show stroke');
    } else {
      $('#image').prop('hidden', true);
      $('#canvas').prop('hidden', false);
      $('#stroke').addClass('active');
      $('#stroke .btn-text').text('Hide stroke');
    }
  });
  $('#start-new').click(onStartNew);
  $('#start').click(onStart);
}

function download(data, filename) {
  var link = document.createElement('a');
  link.href = data;
  link.download = filename;
  link.click();
}

function canvas_auto_resize() {
  console.log("canvas");
  var x = document.getElementById('image-container');
  var h = x.clientHeight;
  var w = x.clientWidth;
  var size = h < w ? h : w;
  annotator.setSize(size, size);
  x = document.getElementById('image');
  x.width = size;
  x.height = size;
  x = document.getElementById('val-panel');
  x.setAttribute("style", "height:" + size + "px");
  x = document.getElementById('ann-panel');
  x.setAttribute("style", "height:" + size + "px");
}

$(document).ready(function () {
  // get config
  readTextFile("static/config.json",
    function(text){
      config = JSON.parse(text);
      allModel = Object.keys(config.models);
      curModelID = 0;
      var key = allModel[curModelID];
      imwidth = config.models[key].output_size;
      imheight = config.models[key].output_size;
      document.getElementById('model-label').textContent = key;
      annotator = new Graph(document, 'canvas');
      canvas_auto_resize();
      init();
    });
});
