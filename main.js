
const tf = require('./tf');
const path = require('path')

const data = require('./data');
const model = require('./model');

async function run(epochs, batchSize, modelSavePath) {
  await data.loadData();

  const {images: trainImages, labels: trainLabels} = data.getTrainData();
  model.summary();

  const validationSplit = 0.15;
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit
  });

  const {images: testImages, labels: testLabels} = data.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log(
      `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

  if (modelSavePath != null) {
    await model.save(tf.io.fileSystem(`${modelSavePath}`));
    console.log(`Saved model to path: ${modelSavePath}`);
  }
}

const savedPath = path.resolve(__dirname, 'model')
run(50, 128, savedPath);
