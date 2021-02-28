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

var graph = null, // canvas manager
    labelgraph = null; // label canvas manager
var currentModel = 0; // current model name
var loading = false; // the image is loading (waiting response)
var image = null, // image data
    label = null; // label image data
    //latent = null, // latent data
    //noise = null; // noise data
var config = null; // config file
var imwidth = 256, // image size
    imheight = 256; // image size
var use_args = false; // [deprecated]
var spinner = new Spinner({ color: '#999' });

var FACE_CATEGORY = ['background', 'skin', 'nose', 'eye glasses', 'eye', 'brow', 'ear', 'mouth', 'upper lip', 'lower lip', 'hair', 'hat', 'ear rings', 'neck', 'cloth'];
var STYLEGAN2_BEDROOM_CATEGORY = ['wall','floor','ceiling','bed','windowpane','table','curtain','painting','lamp','cushion','pillow','flower','light','chandelier','fan','clock'];

var FACE_CATEGORY_COLORS = [
  'rgb(0, 0, 0)', 'rgb(255, 255, 0)', 'rgb(28, 230, 255)',
  'rgb(255, 52, 255)', 'rgb(255, 74, 70)', 'rgb(0, 137, 65)',
  'rgb(0, 111, 166)', 'rgb(163, 0, 89)', 'rgb(255, 219, 229)',
  'rgb(122, 73, 0)', 'rgb(0, 0, 166)', 'rgb(99, 255, 172)',
  'rgb(183, 151, 98)', 'rgb(0, 77, 67)', 'rgb(143, 176, 255)'];
var STYLEGAN2_BEDROOM_COLORS = [
  'rgb(255, 255, 0)', 'rgb(28, 230, 255)',
  'rgb(255, 52, 255)', 'rgb(255, 74, 70)', 'rgb(0, 137, 65)',
  'rgb(0, 111, 166)', 'rgb(163, 0, 89)', 'rgb(255, 219, 229)',
  'rgb(122, 73, 0)', 'rgb(0, 0, 166)', 'rgb(99, 255, 172)',
  'rgb(183, 151, 98)', 'rgb(0, 77, 67)', 'rgb(143, 176, 255)'];

var category = null, category_colors = null;
var name2cat = {
  'Bedroom' : [STYLEGAN2_BEDROOM_CATEGORY, STYLEGAN2_BEDROOM_COLORS],
  'Face' : [FACE_CATEGORY, FACE_CATEGORY_COLORS]};
  

var MODEL_NAMES = []

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

var MAX_LINE_WIDTH = 20;

function setColor(color) {
  graph.setCurrentColor(color);
  $('#color-drop-menu .color-block').css('background-color', color);
  $('#color-drop-menu .color-block').css('border', color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none');
}

function setCategory(color) {
  labelgraph.setCurrentColor(color);
  $('#category-drop-menu .color-block').css('background-color', color);
  $('#category-drop-menu .color-block').css('border', color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none');
}

function setLineWidth(width) {
  graph.setLineWidth(width * 2);
  labelgraph.setLineWidth(width * 2);
  $('#width-label').text(width);
}

function setModel(model) {
  if (loading) return;
  currentModel = model;
  $('#model-label').text(MODEL_NAMES[model]);
  onStart();
}

function setImage(data) {
  setLoading(false);
  if (!data || !data.ok) return;

  if (!image) {
    $('#stroke').removeClass('disabled');
    $('#clear-image').prop('hidden', false);
    $('#clear-label').prop('hidden', false);
    $('#option-buttons').prop('hidden', false);
    $('#option-buttons').prop('hidden', false);
    $('#extra-args').prop('hidden', false);
  }

  image = data.img;
  label = data.label;
  //latent = data.latent;
  //noise = data.noise;
  $('#image').attr('src', image);
  $('#canvas').css('background-image', 'url(' + image + ')');
  $('#label').attr('src', label);
  $('#label-canvas').css('background-image', 'url(' + label + ')');
  graph.setHasImage(true);
  labelgraph.setHasImage(true);
  spinner.spin();
}

function setLoading(isLoading) {
  loading = isLoading;
  graph.setHasImage(false);
  labelgraph.setHasImage(false);
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
  if (graph && !loading) {
    setLoading(true);
    var formData = {
      model: MODEL_NAMES[currentModel],
      image_stroke: graph.getImageData(),
      label_stroke: labelgraph.getImageData()
      //latent: latent,
      //noise: noise
    };
    $.post('stroke', formData, setImage, 'json');
  }
}

function onStartNew() {
  if (loading) return;
  setLoading(true);
  graph.clear();
  labelgraph.clear();
  $.post('new', { model: MODEL_NAMES[currentModel] }, setImage, 'json');
}

function onStart() {
  onStartNew();
  $('#start').prop('hidden', true);
}

function onUpload() {
  // filename = $('#choose').val();
  // if (!filename) {
  //   alert('未选择文件');
  //   return false;
  // }
  // $.get(filename, function(data) {
  //   console.log(data);
  // });
  // return false;
}

function onChooseFile(e) {
  // filename = $('#choose').val();
  // console.log(filename);
  // $('.custom-file-control').content(filename);
  // console.log($('.custom-file-control').after());
}

function init_menu() {
  category_colors.forEach(function (color) {
    $('#color-menu').append(
      '\n<li role="presentation">\n  <div onclick="setColor(\'' +
      color +
      '\')"\n  >\n    <div class="color-block" style="background-color:' +
      color + ';border:' +
      (color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none') +
      '"/>\n  </div>\n</li>');
  });

  category_colors.forEach(function (color, idx) {
    $('#category-menu').append(
      '\n<li role="presentation">\n' +
      ' <div style="float:left;width:100%" onclick="setCategory(\'' +color + '\')">\n' + 
      '   <div class="color-block" style="float:left;background-color:' + color + 
      ';border:' + (color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none') + '"/>\n' +
      '   <div class="semantic-block" >' + 
      category[idx] + '</div>\n</div>\n</li>');
  });
}

function init() {
  init_menu();
  MODEL_NAMES.forEach(function (model, idx) {
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

  setColor('black');
  setCategory(category_colors[0]);
  setLineWidth(MAX_LINE_WIDTH / 2);

  $('#download-sketch').click(function () {
    download(
      graph.getImageData(),
      'sketch_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#download-image').click(function () {
    download(
      image,
      'image_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#download-doodle').click(function () {
    download(
      labelgraph.getImageData(),
      'doodle_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#download-label').click(function () {
    download(
      label,
      'label_' + new Date().format('yyyyMMdd_hhmmss') + '.png');
  });
  $('#clear-image').click(graph.clear);
  $('#clear-label').click(labelgraph.clear);
  $('#submit').click(onSubmit);
  $('#stroke').click(function() {
    var stroke = $('#stroke').hasClass('active');
    if (stroke) {
      $('#image').prop('hidden', false);
      $('#label').prop('hidden', false);
      $('#canvas').prop('hidden', true);
      $('#label-canvas').prop('hidden', true);
      $('#stroke').removeClass('active');
      $('#stroke .btn-text').text('Show stroke');
    } else {
      $('#image').prop('hidden', true);
      $('#label').prop('hidden', true);
      $('#canvas').prop('hidden', false);
      $('#label-canvas').prop('hidden', false);
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

$(document).ready(function () {
  // get config
  readTextFile("static/config.json",
    function(text){
      config = JSON.parse(text);
      MODEL_NAMES = Object.keys(config.models);
      var key = MODEL_NAMES[0];
      category = name2cat[key][0];
      category_colors = name2cat[key][1];
      imwidth = config.models[key].output_size;
      imheight = config.models[key].output_size;
      document.getElementById('model-label').textContent = key;
      graph = new Graph(document, 'canvas');
      labelgraph = new Graph(document, 'label-canvas');
      var x = document.getElementById('image-container');
      var h = x.clientHeight;
      var w = x.clientWidth;
      var size = h < w ? h : w;
      graph.setSize(size, size);
      labelgraph.setSize(size, size);
      x = document.getElementById('image');
      x.width = size;
      x.height = size;
      x = document.getElementById('label');
      x.width = size;
      x.height = size;
      init();
    });
});
