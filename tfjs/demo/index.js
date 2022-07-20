/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

 import * as tf from '@tensorflow/tfjs';
 const MOBILENET_MODEL_PATH = 'http://localhost:5000/web_model/model.json';
 const IMAGENET_CLASSES = ["F", "over60", "18-60", "less18"]
 const IMAGE_SIZE_H = 256;
 const IMAGE_SIZE_W = 128;
 const TOPK_PREDICTIONS = 4;
 const image_size = [256, 128];
 
 function calresizezone(img, radio, v, is_h = false) {
   let dst = new cv.Mat();
   if (is_h) {
     v = Math.floor(radio * v)
     cv.resize(img, dst, new cv.Size(v, image_size[0]), 0, 0, cv.INTER_AREA)
     return dst
   }
   else {
     v = Math.floor(radio * v)
     cv.resize(img, dst, new cv.Size(image_size[1], v), 0, 0, cv.INTER_AREA)
     return dst
   }
 }
 
 function fullpix(v, img_s) {
   a1 = a2 = 0
   if (v < img_s) {
     offset = img_s - v
     if (offset % 2 > 0) {
       a1 = Math.floor((offset - 1) / 2)
       a2 = Math.floor((offset - 1) / 2) + 1
     } else {
       a1 = a2 = Math.floor(offset / 2)
     }
   }
   return {
     a1, a2
   }
 }
 
 function resize_image(image) {
   let { height, width } = image.size();
   if (height > image_size[0] || width > image_size[1]) {
     h_offset = height - image_size[0]
     w_offset = width - image_size[1]
     h_radio = image_size[0] / height
     w_radio = image_size[1] / width
     if (h_offset > 0 || w_offset > 0) {
       if (h_radio < w_radio) {
         image = calresizezone(image, h_radio, width, true)
       }
       else {
         image = calresizezone(image, w_radio, height)
       }
     }
   }
   else if (height < image_size[0] && width < image_size[1]) {
     h_radio = image_size[0] / height
     w_radio = image_size[1] / width
     if (h_radio < w_radio) {
       image = calresizezone(image, h_radio, width, true)
     }
     else {
       image = calresizezone(image, w_radio, height)
     }
   }
   let newSize = image.size();
   let h12 = fullpix(newSize.height, image_size[0])
   let w12 = fullpix(newSize.width, image_size[1])
   let dstImage = new cv.Mat();
   cv.copyMakeBorder(image, dstImage, h12.a1, h12.a2, w12.a1, w12.a2, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 255))
   return dstImage
 }
 
 // tf.setBackend('cpu');
 
 let mobilenet;
 const mobilenetDemo = async () => {
   status('Loading model...');
 
   mobilenet = await tf.loadGraphModel(MOBILENET_MODEL_PATH);
   status('Load done');
   // Warmup the model. This isn't necessary, but makes the first prediction
   // faster. Call `dispose` to release the WebGL memory allocated for the return
   // value of `predict`.
   mobilenet.predict(tf.zeros([1, IMAGE_SIZE_H, IMAGE_SIZE_W, 3])).dispose();
 
   status('');
 
   // Make a prediction through the locally hosted cat.jpg.
   const catElement = document.getElementById('test');
   let img = document.createElement('img');
   if (catElement.complete && catElement.naturalHeight !== 0) {
      predict(img);
      catElement.style.display = '';
   } else {
      catElement.onload = e => {
      img.src = e.target.result;
      predict(img);
      catElement.style.display = '';
     }
   }
 
   document.getElementById('file-container').style.display = '';
 };

 /**
  * Given an image element, makes a prediction through mobilenet returning the
  * probabilities of the top K classes.
  */
 async function predict(imgElement) {
   status('Predicting...');
   const startTime1 = performance.now();
   let startTime2;
   const logits = tf.tidy(() => {
     const cvimg = cv.imread(imgElement);
     const resizedImage = resize_image(cvimg)
     imgElement = document.getElementById('canvasOutput'); 
     let mat2 = new cv.Mat();
     cv.cvtColor(resizedImage, mat2, cv.COLOR_BGRA2RGB, 0);
     cv.imshow('canvasOutput', mat2);
     const tensor = tf.browser.fromPixels(document.getElementById('canvasOutput'))
       .div(255.0).expandDims(0);
 
     startTime2 = performance.now();
     // Make a prediction through mobilenet.
     return mobilenet.predict(tensor);
   });
   // console.log(logits.dataSync())
   // Convert logits to probabilities and class names.
   const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
   const totalTime1 = performance.now() - startTime1;
   const totalTime2 = performance.now() - startTime2;
   status(`Done in ${Math.floor(totalTime1)} ms ` +
     `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);
 
   // Show the classes in the DOM.
   showResults(imgElement, classes);
 }
 
 /**
  * Computes the probabilities of the topK classes given logits by computing
  * softmax to get probabilities and then sorting the probabilities.
  * @param logits Tensor representing the logits from MobileNet.
  * @param topK The number of top predictions to show.
  */
 export async function getTopKClasses(logits, topK) {
   const values = await logits.data();
 
   const valuesAndIndices = [];
   for (let i = 0; i < values.length; i++) {
     valuesAndIndices.push({ value: values[i], index: i });
   }
   const topkValues = new Float32Array(topK);
   const topkIndices = new Int32Array(topK);
   for (let i = 0; i < topK; i++) {
     topkValues[i] = valuesAndIndices[i].value;
     topkIndices[i] = valuesAndIndices[i].index;
   }
   status(topkValues)
   const topClassesAndProbs = [];
   for (let i = 0; i < topkIndices.length; i++) {
     topClassesAndProbs.push({
       className: IMAGENET_CLASSES[topkIndices[i]],
       probability: topkValues[i]
     })
   }
   return topClassesAndProbs;
 }
 
 function showResults(imgElement, classes) {
   const predictionContainer = document.createElement('div');
   predictionContainer.className = 'pred-container';
 
   const imgContainer = document.createElement('div');
   imgContainer.appendChild(imgElement);
   predictionContainer.appendChild(imgContainer);
 
   const probsContainer = document.createElement('div');
   for (let i = 0; i < classes.length; i++) {
     const row = document.createElement('div');
     row.className = 'row';
 
     const classElement = document.createElement('div');
     classElement.className = 'cell';
     classElement.innerText = classes[i].className;
     row.appendChild(classElement);
 
     const probsElement = document.createElement('div');
     probsElement.className = 'cell';
     probsElement.innerText = classes[i].probability.toFixed(3);
     row.appendChild(probsElement);
 
     probsContainer.appendChild(row);
   }
   predictionContainer.appendChild(probsContainer);
 
   predictionsElement.insertBefore(
     predictionContainer, predictionsElement.firstChild);
 }
 
 const filesElement = document.getElementById('files');
 filesElement.addEventListener('change', evt => {
   let files = evt.target.files;
   // Display thumbnails & issue call to predict each image.
   for (let i = 0, f; f = files[i]; i++) {
     // Only process image files (skip non image files)
     if (!f.type.match('image.*')) {
       continue;
     }
     let reader = new FileReader();
     reader.onload = e => {
       // Fill the image & call predict.
       let img = document.createElement('img');
       img.src = e.target.result;
       img.onload = () => predict(img);
     };
    // Read in the image file as a data URL.
     reader.readAsDataURL(f);
   }
 });
 
 const demoStatusElement = document.getElementById('status');
 const status = msg => demoStatusElement.innerText = msg;
 const predictionsElement = document.getElementById('predictions');
 
 mobilenetDemo();
 