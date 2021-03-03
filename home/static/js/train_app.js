'use strict';

/*** Read a text file from disk. ***/
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

/*** Format download file name. ***/
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
  }
  return format;
};


var annotator = null; // canvas manager
var loading = false; // the image is loading (waiting response)
var image = null; // image data
var images = [], anns = [];
var config = null; // config file
var imwidth = 256, // image size
    imheight = 256; // image size
var use_args = false; // [deprecated]
var spinner = new Spinner({ color: '#999' });
var allModel = [];
var curModelID = null;
var category = ['background'];

var COLORS = [
  'rgb(0, 0, 0)', 'rgb(255, 255, 0)', 'rgb(28, 230, 255)',
  'rgb(255, 52, 255)', 'rgb(255, 74, 70)', 'rgb(0, 137, 65)',
  'rgb(0, 111, 166)', 'rgb(163, 0, 89)', 'rgb(255, 219, 229)',
  'rgb(122, 73, 0)', 'rgb(0, 0, 166)', 'rgb(99, 255, 172)',
  'rgb(183, 151, 98)', 'rgb(0, 77, 67)', 'rgb(143, 176, 255)'];
var MAX_LINE_WIDTH = 20;


/*** Set the semanticcategory of the pen. ***/
function setCategory(color) {
  annotator.setCurrentColor(color);
  $('#category-drop-menu .color-block').css('background-color', color);
  $('#category-drop-menu .color-block').css('border', color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none');
}

/*** Set the line width of the pen. ***/
function setLineWidth(width) {
  annotator.setLineWidth(width * 2);
  $('#width-label').text(width);
}

/*** Set the current model. ***/
function setModel(model) {
  if (loading) return;
  curModelID = model;
  $('#model-label').text(allModel[model]);
  onStart();
}

/*** Set the generated image. ***/
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

function imageDiv(image, ann, width, padding, overlap) {
  // put images in a div
  var div = document.createElement("div");
  var s = "position: relative; top: 10px;";
  s += "left: " + padding + "px;"
    
  div.setAttribute("style", s);

  // original image
  var img = document.createElement('img');
  img.src = image;
  img.setAttribute("style", "position: relative;");
  img.width = width;
  div.appendChild(img);

  // annotation
  var img = document.createElement('img');
  img.src = ann;
  img.width = width;
  img.opacity = 0.5;
  if (overlap)
    var s = "position: absolute; left: 0; right: 0;";
  else
    var s = "position: relative;";
  img.setAttribute("style", s);
  div.appendChild(img);

  return div;
}

/*** Put the annotation onto the panel. ***/
function setAnn() {
  setLoading(false);
  var ncols = 3, padding = 10;
  var panel = document.getElementById("ann-panel");
  panel.innerHTML = '';
  var size = panel.offsetWidth / ncols - padding * (ncols - 1);

  for (var i = 0; i < images.length; i ++) {
    var col = i % ncols;
    panel.appendChild(imageDiv(
      images[images.length - 1],
      anns[anns.length - 1],
      size,
      col * padding,
      true));
  }
  annotator.setHasImage(true);
  spinner.spin();
}

/*** Set the loading status of image. ***/
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

/*** Upload the annotation to the server. ***/
function onAnnotate() {
  console.log("on annotate");
  if (annotator && !loading) {
    setLoading(true);
    var ann = annotator.getImageData();
    var formData = {
      model: allModel[curModelID],
      ann: ann,
    };
    images.push(image);
    anns.push(ann);
    $.post('train/ann', formData, setAnn, 'json');
  }
}

function onTrain() {

}

function setValidation(data) {
  console.log(data);
  var ncols = 4, ngroups = 2, padding = 10;
  var panel = document.getElementById("val-panel");
  panel.innerHTML = '';
  var size = panel.offsetWidth / ncols - padding;
  for (var i = 0; i < data.images.length; i ++) {
    var col = i % ngroups;
    panel.appendChild(imageDiv(
      data.images[i],
      data.labels[i],
      size,
      col * padding,
      false));
  }
}

function onValidate() {
  $.post(
    'train/val',
    {model : allModel[curModelID]},
    setValidation,
    'json');
}

/*** Generate a new image. ***/
function onStartNew() {
  if (loading) return;
  setLoading(true);
  annotator.clear();
  $.post('train/new',
    { model: allModel[curModelID] },
    setImage, 'json');
  console.log({ model: allModel[curModelID] });
}

/*** Generate a new image for the first time. ***/
function onStart() {
  onStartNew();
  $('#start').prop('hidden', true);
}

/*** Remove all the image and annotations in the panel. ***/
function onClearAnn() {
  // send a message to the server
  $.post('train/clear', {model : allModel[curModelID]});
}

/*** The user add a new category for labeling. ***/
function addCategory() {
  var mod = document.getElementById('new-cat-mod');
  var name = document.getElementById('category-name').value;
  var idx = category.length;
  if (name.length == 0) return;

  category.push(name);

  document.getElementById('category-menu').removeChild(mod);
  $('#category-menu').append('\n<li role="presentation">\n' +
  ' <div style="float:left;width:100%" onclick="setCategory(\'' + 
  COLORS[idx] + '\')">\n' + 
  '  <div class="color-block" style="float:left;background-color:' + 
  COLORS[idx] + ';border:' + colorStyle(COLORS[idx]) + '"/>\n' +
  '   <div class="semantic-block" >' + 
      name + '</div>\n</div>\n</li>');

  menuNewCategory();
}

/*** A utility function. ***/
function colorStyle(color) {
  return color == 'white' ? 'solid 1px rgba(0, 0, 0, 0.2)' : 'none';
}

/*** Add a new category textarea to the menu. ***/
function menuNewCategory() {
  var next = category.length;
  $('#category-menu').append(
    '\n<li role="presentation" id="new-cat-mod">\n' +
    ' <div style="float:left;width:100%" onclick="addCategory(\'' + 
          COLORS[next] + '\')">\n' + 
    '   <div class="color-block" style="float:left;background-color:' + 
    COLORS[next] + ';border:' + colorStyle(COLORS[next]) + '"/>\n' + 
    '    <textarea id="category-name" rows="1" cols="16"></textarea>' + 
    '    <div class="add-block">new</div>\n</div>\n</li>');
}

function init() {
  // Add category menu items
  category.forEach(function (c, idx) {
    $('#category-menu').append(
      '\n<li role="presentation">\n' +
      ' <div style="float:left;width:100%" onclick="setCategory(\'' + 
      COLORS[idx] + '\')">\n' + 
      '  <div class="color-block" style="float:left;background-color:' + 
      COLORS[idx] + ';border:' + colorStyle(COLORS[idx]) + '"/>\n' +
      '   <div class="semantic-block" >' + 
          c + '</div>\n</div>\n</li>');
  });
  menuNewCategory();
  setCategory(COLORS[0]); // default category

  // Add model menu items
  allModel.forEach(function (model, idx) {
    $('#model-menu').append(
      '<li role="presentation">\n' +
      '  <div class="dropdown-item model-item" onclick="setModel(' + idx + ')">' +
      model +
      '  </div>\n' +
      '</li>\n'
    );
  });

  // Init slider
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

  setLineWidth(MAX_LINE_WIDTH / 2);

  /*
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
  */

  $('#clear').click(annotator.clear);
  $('#clear-ann').click(onClearAnn);
  $('#ann-btn').click(onAnnotate);
  $('#train-btn').click(onTrain);
  $('#val-btn').click(onValidate);

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
  //x = document.getElementById('val-panel');
  //x.setAttribute("style", "height:" + size + "px");
  //x = document.getElementById('ann-panel');
  //x.setAttribute("style", "height:" + size + "px");
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
