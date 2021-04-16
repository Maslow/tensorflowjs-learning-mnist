
const tf = require('./tf');
const path = require('path')
const fs = require('fs')
const savedPath = path.resolve(__dirname, 'model/model.json')
const sharp = require('sharp');


async function main() {
  const pic_no = 5
  const model = await tf.loadLayersModel(tf.io.fileSystem(savedPath))
  model.summary()

  const buffer = fs.readFileSync(`./${pic_no}.png`)
  const {data, info} = await sharp(buffer).resize(28, 28)
    .greyscale()
    .normalize()
    .toColorspace('b-w')
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })

  // 将图像像素数据转换为 float32 数组
  const pixels = new Float32Array(info.size)
  for (let i = 0; i < data.byteLength; i++) {
    pixels[i] = data.readUInt8(i)
  }
  
  await sharp(data, { raw: { width: info.width, height: info.height, channels: info.channels } })
    .toFile(`./temp/${pic_no}-sm.png`)
  
  // 值化：将灰度值转为 0~1 之间的数字
  // 反转：输入图为白底黑字，而训练时用的黑底白字，需要反转
  const values = pixels.map(v => 1 - v/255)
  const input = tf.tensor4d(values, [1, 28, 28, 1])
  // const input = tf.tensor3d(values, [28, 28, 1]).reshape([-1, 28, 28, 1])

  const ret = model.predict(input).dataSync()
  console.log(ret)
  let max = -1, index = -1
  for (let i = 0; i < ret.length; i++) {
    if (max < ret[i]) {
      max = ret[i]
      index = i
    }
  }

  console.log({result: index, confident: max})
}

main()