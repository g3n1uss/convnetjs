 // ------------------------
  // BEGIN MNIST SPECIFIC STUFF
  // ------------------------
  classes_txt = ['0','1','2','3','4','5','6','7','8','9'];
  var dataset_name = "mnist";
  var num_batches = 21; // 20 training batches, 1 test
  var test_batch = 20;
  var num_samples_per_batch = 3000;
  var image_dimension = 28;
  var image_channels = 1;
  var use_validation_data = true;
  var random_flip = false;
  var random_position = false;
  // ------------------------
  // END MNIST SPECIFIC STUFF
  // ------------------------

var data_img_elts = new Array(num_batches);
var img_data = new Array(num_batches);
var loaded = new Array(num_batches);
var loaded_train_batches = [];
var lossGraph = new cnnvis.Graph();
var xLossWindow = new cnnutil.Window(100);
var wLossWindow = new cnnutil.Window(100);
var trainAccWindow = new cnnutil.Window(100);
var valAccWindow = new cnnutil.Window(100);
var testAccWindow = new cnnutil.Window(50, 1);
var step_num = 0;

// keep 'images_per_page' images per page in the prediction section (Example predictions on Test set)
var images_per_page=20;

// int main

// use jQuery to evaluate everything inside this function after the page is loaded
// first create all the elements within the document, then assign them values
$(window).load(function() {

  net = new convnetjs.Net();
  document.getElementById('newnet').value=net.conf_string

  // set up the trainer
  trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:20, l2_decay:0.001});
  

  // read off the rest of the parameters as defaults from the trainer
  // if the configuration has been changed - update it
  update_net_param_display();

  // what is this loop doing? just setting all the elements of loaded to false? why?
  for(var k=0;k<loaded.length;k++) { loaded[k] = false; }

  //load zeroth batch (training batch) and 21st (testing), the rest 19 training batches will be loaded 
  //while the network is training on the zeroth one in 'sample_training_instance()'
  load_data_batch(0);

  // test_batch=20 (defined at mnist page, get rid of this variable?)
  load_data_batch(test_batch);

  // run the function 'load_and_step' though this auxiliary function, which checks wheather 
  // the batches are loaded
  start_fun();
});

// load parameters as the values defined in the trainer. If this is the first run - read off 
// from the default parameters of the trainer
var update_net_param_display = function() {
  document.getElementById('train_method').value = trainer.method;
  document.getElementById('lr_input').value = trainer.learning_rate;
  document.getElementById('momentum_input').value = trainer.momentum;
  document.getElementById('batch_size_input').value = trainer.batch_size;
  document.getElementById('decay_input').value = trainer.l2_decay;
  document.getElementById('epsilon_input').value = trainer.eps;
}

// load the dataset with JS in background
var load_data_batch = function(batch_num) {
  data_img_elts[batch_num] = new Image();
  var data_img_elt = data_img_elts[batch_num];
  // async batch load, happens after the path to batch is set (function following this one) whenever 
  // the broweser has some time, this time is allocated by function 'start_fun' via 'setTimeout'
  data_img_elt.onload = function() { 
    var data_canvas = document.createElement('canvas');
    data_canvas.width = data_img_elt.width;
    data_canvas.height = data_img_elt.height;
    var data_ctx = data_canvas.getContext("2d");
    data_ctx.drawImage(data_img_elt, 0, 0); // copy it over... bit wasteful :(
    img_data[batch_num] = data_ctx.getImageData(0, 0, data_canvas.width, data_canvas.height);
    loaded[batch_num] = true;
    if(batch_num < test_batch) { loaded_train_batches.push(batch_num); }
    console.log('finished loading data batch ' + batch_num);
  };
  // set the path to the batch, whenever there is a free time the data will be loaded  by 
  // the above function. The broweser decides if there is a free time in 'start_fun', that 
  // function gives a brower time to load data via setTimeout, it allows the program to 
  // proceed (call load_and_step) if both batches are loaded
  data_img_elt.src = dataset_name + "/" + dataset_name + "_batch_" + batch_num + ".png";
}

// run function 'load_and_step' after loading both training and testing batches
var start_fun = function() {
  if(loaded[0] && loaded[test_batch]) { 
    console.log('starting!'); 
    setInterval(load_and_step, 0); // lets go!
  }
  else { setTimeout(start_fun, 200); } // keep checking
}

// load a training image and train on it with the network
var paused = true;
// train unless paused
var load_and_step = function() {
  if(paused) return; 
  // sample has volume object and the corresponding label
  var sample = sample_training_instance();
  step(sample); // process this image
  
  //setTimeout(load_and_step, 0); // schedule the next iteration
}

// generate a training sample
var sample_training_instance = function() {
  // find an unloaded batch

  // 'loaded_train_batches' contains loaded batches pushed in 'load_data_batch' via async load
  var bi = Math.floor(Math.random()*loaded_train_batches.length);
  var b = loaded_train_batches[bi];
  var k = Math.floor(Math.random()*num_samples_per_batch); // pick a random sample within the batch
  var n = b*num_samples_per_batch+k;

  // load more batches over time
  if(step_num%(2 * num_samples_per_batch)===0 && step_num>0) {
    // after the current sample is completely worked out, 'step_num' is incremented in function 'step()' 
    // in the web version '2*num_samples_per_batch' is replaced by 5000, it does not change much?
    for(var i=0;i<num_batches;i++) {
      if(!loaded[i]) {
        // load it
        load_data_batch(i);
        break; // okay for now
      }
    }
  }

  // fetch the appropriate row of the training image and reshape into a Vol
  // p is data in the RGBA format, array of lenth 9408000: 3000 samples per batch, 28x28 sample, 4 channels RGBA
  var p = img_data[b].data;
  // create object Vol: it has two arrays w and dw of length 784 filled with zeros at this point
  var x = new convnetjs.Vol(image_dimension,image_dimension,image_channels,0.0);
  var W = image_dimension*image_dimension;
  // the following part is very different from the web version, but does not make much difference
  /*
  var j=0;
  for(var dc=0;dc<image_channels;dc++) {
    var i=0;
    for(var xc=0;xc<image_dimension;xc++) {
      for(var yc=0;yc<image_dimension;yc++) {
        var ix = ((W * k) + i) * 4 + dc;
        x.set(yc,xc,dc,p[ix]/255.0-0.5);
        i++;
      }
    }
  }

  if(random_position){
    var dx = Math.floor(Math.random()*5-2);
    var dy = Math.floor(Math.random()*5-2);
    x = convnetjs.augment(x, image_dimension, dx, dy, false); //maybe change position
  }

  if(random_flip){
    x = convnetjs.augment(x, image_dimension, 0, 0, Math.random()<0.5); //maybe flip horizontally
  }*/

  // code from the web version
  
  // that is how we fill volume object V from data p, 
  // it is pretty random and seems like have a potential problem
  for(var i=0;i<W;i++) {
    var ix = ((W * k) + i) * 4;// k is a random sample within the batch
    // 4 is coming from the fact that we deal with RGBA format, it has 4 channels

    // this part is dangerous: k is a random number between 0 and 3000,
    // on the other hand the length of p is 3000*W*4, so we have a potential index out of bound ix
    x.w[i] = p[ix]/255.0;
  }
  x = convnetjs.augment(x, 24);



  var isval = use_validation_data && n%10===0 ? true : false;
  return {x:x, label:labels[n], isval:isval};
}

// train on the picked sample and visualize the process
var step = function(sample) {
  // read off the data and the label
  var x = sample.x;
  var y = sample.label;
  
  // if sample is testing test and ge out, else - train on it
  if(sample.isval) {
    // use x to build our estimate of validation error
    net.forward(x);
    var yhat = net.getPrediction();
    var val_acc = yhat === y ? 1.0 : 0.0;
    valAccWindow.add(val_acc);
    return; // get out
  }

  // train on it with network
  var stats = trainer.train(x, y);
  var lossx = stats.cost_loss;
  var lossw = stats.l2_decay_loss;

  // keep track of stats such as the average training error and loss
  var yhat = net.getPrediction();
  var train_acc = yhat === y ? 1.0 : 0.0;
  xLossWindow.add(lossx);
  wLossWindow.add(lossw);
  trainAccWindow.add(train_acc);

  // print training status
  var train_elt = document.getElementById("trainstats");
  train_elt.innerHTML = '';
  var t = 'Forward time per example: ' + stats.fwd_time + 'ms';
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Backprop time per example: ' + stats.bwd_time + 'ms';
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Classification loss: ' + f2t(xLossWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'L2 Weight decay loss: ' + f2t(wLossWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Training accuracy: ' + f2t(trainAccWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Validation accuracy: ' + f2t(valAccWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Examples seen: ' + step_num;
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));

  // visualize activations
  if(step_num % 100 === 0) {
    var vis_elt = document.getElementById("visnet");
    visualize_activations(net, vis_elt);
  }

  // log progress to graph, (full loss)
  if(step_num % 200 === 0) {
    var xa = xLossWindow.get_average();
    var xw = wLossWindow.get_average();
    if(xa >= 0 && xw >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
      lossGraph.add(step_num, xa + xw);
      lossGraph.drawSelf(document.getElementById("lossgraph"));
    }
  }

  // run prediction on test set
  if((step_num % 100 === 0 && step_num > 0) || step_num===100) {
    test_predict();
  }
  step_num++;
}

// sample a random testing instance
var sample_test_instance = function() {

  var b = test_batch;
  var k = Math.floor(Math.random()*num_samples_per_batch);
  var n = b*num_samples_per_batch+k;

  var p = img_data[b].data;
  var x = new convnetjs.Vol(image_dimension,image_dimension,image_channels,0.0);
  var W = image_dimension*image_dimension;
  // the following piece does not work and different from the web version 
  /*var j=0;
  for(var dc=0;dc<image_channels;dc++) {
    var i=0;
    for(var xc=0;xc<image_dimension;xc++) {
      for(var yc=0;yc<image_dimension;yc++) {
        var ix = ((W * k) + i) * 4 + dc;
        x.set(yc,xc,dc,p[ix]/255.0-0.5);
        i++;
      }
    }
  }

  // distort position and maybe flip
  var xs = [];
  
  if (random_flip || random_position){
    for(var k=0;k<6;k++) {
      var test_variation = x;
      if(random_position){
        var dx = Math.floor(Math.random()*5-2);
        var dy = Math.floor(Math.random()*5-2);
        test_variation = convnetjs.augment(test_variation, image_dimension, dx, dy, false);
      }
      
      if(random_flip){
        test_variation = convnetjs.augment(test_variation, image_dimension, 0, 0, Math.random()<0.5); 
      }

      xs.push(test_variation);
    }
  }else{
    xs.push(x, image_dimension, 0, 0, false); // push an un-augmented copy
  }*/
  for(var i=0;i<W;i++) {
    var ix = ((W * k) + i) * 4;
    x.w[i] = p[ix]/255.0;
  }
  var xs = [];
  for(var i=0;i<4;i++) {
    xs.push(convnetjs.augment(x, 24));
  }
  
  // return multiple augmentations, and we will average the network over them
  // to increase performance
  return {x:xs, label:labels[n]};
}


// visualize and test network


var maxmin = cnnutil.maxmin;
var f2t = cnnutil.f2t;

// elt is the element to add all the canvas activation drawings into
// A is the Vol() to use
// scale is a multiplier to make the visualizations larger. Make higher for larger pictures
// if grads is true then gradients are used instead
var draw_activations = function(elt, A, scale, grads) {

  var s = scale || 2; // scale
  var draw_grads = false;
  if(typeof(grads) !== 'undefined') draw_grads = grads;

  // get max and min activation to scale the maps automatically
  var w = draw_grads ? A.dw : A.w;
  var mm = maxmin(w);

  // create the canvas elements, draw and add to DOM
  for(var d=0;d<A.depth;d++) {

    var canv = document.createElement('canvas');
    canv.className = 'actmap';
    var W = A.sx * s;
    var H = A.sy * s;
    canv.width = W;
    canv.height = H;
    var ctx = canv.getContext('2d');
    var g = ctx.createImageData(W, H);

    for(var x=0;x<A.sx;x++) {
      for(var y=0;y<A.sy;y++) {
        if(draw_grads) {
          var dval = Math.floor((A.get_grad(x,y,d)-mm.minv)/mm.dv*255);
        } else {
          var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);  
        }
        for(var dx=0;dx<s;dx++) {
          for(var dy=0;dy<s;dy++) {
            var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
            for(var i=0;i<3;i++) { g.data[pp + i] = dval; } // rgb
            g.data[pp+3] = 255; // alpha channel
          }
        }
      }
    }
    ctx.putImageData(g, 0, 0);
    elt.appendChild(canv);
  }  
}

var draw_activations_COLOR = function(elt, A, scale, grads) {

  var s = scale || 2; // scale
  var draw_grads = false;
  if(typeof(grads) !== 'undefined') draw_grads = grads;

  // get max and min activation to scale the maps automatically
  var w = draw_grads ? A.dw : A.w;
  var mm = maxmin(w);

  var canv = document.createElement('canvas');
  canv.className = 'actmap';
  var W = A.sx * s;
  var H = A.sy * s;
  canv.width = W;
  canv.height = H;
  var ctx = canv.getContext('2d');
  var g = ctx.createImageData(W, H);
  for(var d=0;d<3;d++) {
    for(var x=0;x<A.sx;x++) {
      for(var y=0;y<A.sy;y++) {
        if(draw_grads) {
          var dval = Math.floor((A.get_grad(x,y,d)-mm.minv)/mm.dv*255);
        } else {
          var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);  
        }
        for(var dx=0;dx<s;dx++) {
          for(var dy=0;dy<s;dy++) {
            var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
            g.data[pp + d] = dval;
            if(d===0) g.data[pp+3] = 255; // alpha channel
          }
        }
      }
    }
  }
  ctx.putImageData(g, 0, 0);
  elt.appendChild(canv);
}

var visualize_activations = function(net, elt) {

  // clear the element
  elt.innerHTML = "";

  // show activations in each layer
  var N = net.layers.length;
  for(var i=0;i<N;i++) {
    var L = net.layers[i];

    var layer_div = document.createElement('div');

    // visualize activations
    var activations_div = document.createElement('div');
    activations_div.appendChild(document.createTextNode('Activations:'));
    activations_div.appendChild(document.createElement('br'));
    activations_div.className = 'layer_act';
    var scale = 2;
    if(L.layer_type==='softmax' || L.layer_type==='fc') scale = 10; // for softmax
    
    // HACK to draw in color in input layer
    if(i===0) {
      draw_activations_COLOR(activations_div, L.out_act, scale);
      draw_activations_COLOR(activations_div, L.out_act, scale, true);

      /*
      // visualize positive and negative components of the gradient separately
      var dd = L.out_act.clone();
      var ni = L.out_act.w.length;
      for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq > 0 ? dwq : 0.0; }
      draw_activations_COLOR(activations_div, dd, scale);
      for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq < 0 ? -dwq : 0.0; }
      draw_activations_COLOR(activations_div, dd, scale);
      */

      /*
      // visualize what the network would like the image to look like more
      var dd = L.out_act.clone();
      var ni = L.out_act.w.length;
      for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] -= 20*dwq; }
      draw_activations_COLOR(activations_div, dd, scale);
      */

      /*
      // visualize gradient magnitude
      var dd = L.out_act.clone();
      var ni = L.out_act.w.length;
      for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq*dwq; }
      draw_activations_COLOR(activations_div, dd, scale);
      */

    } else {
      draw_activations(activations_div, L.out_act, scale);
    } 

    // visualize data gradients
    if(L.layer_type !== 'softmax' && L.layer_type !== 'input' ) {
      var grad_div = document.createElement('div');
      grad_div.appendChild(document.createTextNode('Activation Gradients:'));
      grad_div.appendChild(document.createElement('br'));
      grad_div.className = 'layer_grad';
      var scale = 2;
      if(L.layer_type==='softmax' || L.layer_type==='fc') scale = 10; // for softmax
      draw_activations(grad_div, L.out_act, scale, true);
      activations_div.appendChild(grad_div);
    }

    // visualize filters if they are of reasonable size
    if(L.layer_type === 'conv') {
      var filters_div = document.createElement('div');
      if(L.filters[0].sx>3) {
        // actual weights
        filters_div.appendChild(document.createTextNode('Weights:'));
        filters_div.appendChild(document.createElement('br'));
        for(var j=0;j<L.filters.length;j++) {
          // HACK to draw in color for first layer conv filters
          if(i===1) {
            draw_activations_COLOR(filters_div, L.filters[j], 2);
          } else {
            filters_div.appendChild(document.createTextNode('('));
            draw_activations(filters_div, L.filters[j], 2);
            filters_div.appendChild(document.createTextNode(')'));
          }
        }
        // gradients
        filters_div.appendChild(document.createElement('br'));
        filters_div.appendChild(document.createTextNode('Weight Gradients:'));
        filters_div.appendChild(document.createElement('br'));
        for(var j=0;j<L.filters.length;j++) {
          if(i===1) { draw_activations_COLOR(filters_div, L.filters[j], 2, true); }
          else {
            filters_div.appendChild(document.createTextNode('('));
            draw_activations(filters_div, L.filters[j], 2, true);
            filters_div.appendChild(document.createTextNode(')'));
          }
        }
      } else {
        filters_div.appendChild(document.createTextNode('Weights hidden, too small'));
      }
      activations_div.appendChild(filters_div);
    }
    layer_div.appendChild(activations_div);

    // print some stats on left of the layer
    layer_div.className = 'layer ' + 'lt' + L.layer_type;
    var title_div = document.createElement('div');
    title_div.className = 'ltitle'
    var t = L.layer_type + ' (' + L.out_sx + 'x' + L.out_sy + 'x' + L.out_depth + ')';
    title_div.appendChild(document.createTextNode(t));
    layer_div.appendChild(title_div);

    if(L.layer_type==='conv') {
      var t = 'filter size ' + L.filters[0].sx + 'x' + L.filters[0].sy + 'x' + L.filters[0].depth + ', stride ' + L.stride;
      layer_div.appendChild(document.createTextNode(t));
      layer_div.appendChild(document.createElement('br'));
    }
    if(L.layer_type==='pool') {
      var t = 'pooling size ' + L.sx + 'x' + L.sy + ', stride ' + L.stride;
      layer_div.appendChild(document.createTextNode(t));
      layer_div.appendChild(document.createElement('br'));
    }

    // find min, max activations and display them
    var mma = maxmin(L.out_act.w);
    var t = 'max activation: ' + f2t(mma.maxv) + ', min: ' + f2t(mma.minv);
    layer_div.appendChild(document.createTextNode(t));
    layer_div.appendChild(document.createElement('br'));

    var mma = maxmin(L.out_act.dw);
    var t = 'max gradient: ' + f2t(mma.maxv) + ', min: ' + f2t(mma.minv);
    layer_div.appendChild(document.createTextNode(t));
    layer_div.appendChild(document.createElement('br'));

    // number of parameters
    if(L.layer_type==='conv' || L.layer_type==='local') {
      var tot_params = L.sx*L.sy*L.in_depth*L.filters.length + L.filters.length;
      var t = 'parameters: ' + L.filters.length + 'x' + L.sx + 'x' + L.sy + 'x' + L.in_depth + '+' + L.filters.length + ' = ' + tot_params;
      layer_div.appendChild(document.createTextNode(t));
      layer_div.appendChild(document.createElement('br'));
    }
    if(L.layer_type==='fc') {
      var tot_params = L.num_inputs*L.filters.length + L.filters.length;
      var t = 'parameters: ' + L.filters.length + 'x' + L.num_inputs + '+' + L.filters.length + ' = ' + tot_params;
      layer_div.appendChild(document.createTextNode(t));
      layer_div.appendChild(document.createElement('br'));
    }

    // css madness needed here...
    var clear = document.createElement('div');
    clear.className = 'clear';
    layer_div.appendChild(clear);

    elt.appendChild(layer_div);
  }
}



// evaluate current network on test set and visualize predictions
var test_predict = function() {
  var num_classes = net.layers[net.layers.length-1].out_depth;

  document.getElementById('testset_acc').innerHTML = '';
  var num_total = 0;
  var num_correct = 0;

  // grab a random test image
  for(num=0;num<4;num++) {
    var sample = sample_test_instance();
    var y = sample.label;  // ground truth label

    // forward prop it through the network
    var aavg = new convnetjs.Vol(1,1,num_classes,0.0);
    // ensures we always have a list, regardless if above returns single item or list
    var xs = [].concat(sample.x);
    var n = xs.length;
    for(var i=0;i<n;i++) {
      var a = net.forward(xs[i]);
      aavg.addFrom(a);
    }
    var preds = [];
    for(var k=0;k<aavg.w.length;k++) { preds.push({k:k,p:aavg.w[k]}); }
    preds.sort(function(a,b){return a.p<b.p ? 1:-1;});
    
    var correct = preds[0].k===y;
    if(correct) num_correct++;
    num_total++;

    var div = document.createElement('div');
    div.className = 'testdiv';

    // draw the image into a canvas
    draw_activations_COLOR(div, xs[0], 2); // draw Vol into canv

    // add predictions
    var probsdiv = document.createElement('div');
    
    var t = '';
    for(var k=0;k<3;k++) {
      var col = preds[k].k===y ? 'rgb(85,187,85)' : 'rgb(187,85,85)';
      t += '<div class=\"pp\" style=\"width:' + Math.floor(preds[k].p/n*100) + 'px; background-color:' + col + ';\">' + classes_txt[preds[k].k] + '</div>'
    }
    probsdiv.innerHTML = t;
    probsdiv.className = 'probsdiv';
    div.appendChild(probsdiv);

    // add it into DOM
    $(div).prependTo($("#testset_vis")).hide().fadeIn('slow').slideDown('slow');
    // keep always 'images_per_page' images
    // if there are more than 'images_per_page' pictures - remove the last one
    if($("#testset_vis")[0].childElementCount>images_per_page) {
      var list=document.getElementById("testset_vis");
      list.removeChild(list.childNodes[images_per_page]);
    }
  }
  testAccWindow.add(num_correct/num_total);
  $("#testset_acc").text('test accuracy based on last 200 test images: ' + testAccWindow.get_average());  
}

// user settings

// set the trainer parameters
var setTrainerParams = function(){
  trainer.learning_rate = parseFloat(document.getElementById("lr_input").value);
  trainer.momentum = parseFloat(document.getElementById("momentum_input").value);
  trainer.batch_size = parseFloat(document.getElementById("batch_size_input").value);
  trainer.l2_decay = parseFloat(document.getElementById("decay_input").value);
  trainer.eps = parseFloat(document.getElementById("epsilon_input").value);
}

// pause
var toggle_pause = function() {
  paused = !paused;
  var btn = document.getElementById('buttontp');
  if(paused) { btn.value = 'resume' }
  else { btn.value = 'pause'; }
}
var dump_json = function() {
  document.getElementById("dumpjson").value = JSON.stringify(this.net.toJSON());
}
var clear_graph = function() {
  lossGraph = new cnnvis.Graph(); // reinit graph too 
}

var load_from_json = function() {
  var jsonString = document.getElementById("dumpjson").value;
  var json = JSON.parse(jsonString);
  net = new convnetjs.Net();
  net.fromJSON(json);
  reset_all();
}

var load_pretrained = function() {
  $.getJSON(dataset_name + "_snapshot.json", function(json){
    net = new convnetjs.Net();
    net.fromJSON(json);
    trainer.learning_rate = 0.0001;
    trainer.momentum = 0.9;
    trainer.batch_size = 2;
    trainer.l2_decay = 0.00001;
    reset_all();
  });
}

var change_net = function() {
  // read off the configuration of the network and parse it
  // layers are separated by line breaks
  var newlayerslist=this.newnet.value.split( "\n" );
  // cast the configuration into the array
  var newlayersconf=[];
  for (var i=0;i<newlayerslist.length;i++){
    newlayersconf.push(eval('({'+newlayerslist[i]+'})'));
  }
  // the following code will translate the configuration of layers as simple array 'layers_conf'' into the set of objects
  net.conf_string=this.newnet.value;
  net.makeLayers(newlayersconf);
  reset_all();
}

var reset_all = function() {
  // reinit trainer
  trainer = new convnetjs.SGDTrainer(net, {learning_rate:trainer.learning_rate, momentum:trainer.momentum, batch_size:trainer.batch_size, l2_decay:trainer.l2_decay});
  setTrainerParams();
  // reinit windows that keep track of val/train accuracies
  xLossWindow.reset();
  wLossWindow.reset();
  trainAccWindow.reset();
  valAccWindow.reset();
  testAccWindow.reset();
  lossGraph = new cnnvis.Graph(); // reinit graph too
  step_num = 0;
}
