const fs = require("fs");
const tf = require("@tensorflow/tfjs-node-gpu");

const img2x = (imgPath) => {
  const buffer = fs.readFileSync(imgPath);
  //清除数据
  return tf.tidy(() => {
    //把图片转成tensor
    const imgt = tf.node.decodeImage(new Uint8Array(buffer), 3);
    //调整图片大小
    const imgResize = tf.image.resizeBilinear(imgt, [224, 224]);
    //归一化
    return imgResize
      .toFloat()
      .sub(255 / 2)
      .div(255 / 2)
      .reshape([1, 224, 224, 3]);
  });
};

const getData = async (traindir, output, writeClasses = true) => {
  let classes = fs.readdirSync(traindir, "utf-8");
  if (writeClasses) {
    fs.writeFileSync(`../${output}/classes.json`, JSON.stringify(classes));
  }
  const data = [];
  classes.forEach((dir, dirIndex) => {
    fs.readdirSync(`${traindir}/${dir}`)
      .filter((n) => n.match(/jpg$/))
      .slice(0, 1000)
      .forEach((filename) => {
        const imgPath = `${traindir}/${dir}/${filename}`;

        data.push({ imgPath, dirIndex });
      });
  });

  console.log("data:", data);

  //打乱训练顺序，提高准确度
  tf.util.shuffle(data);

  const ds = tf.data.generator(function* () {
    const count = data.length;
    const batchSize = 32;
    for (let start = 0; start < count; start += batchSize) {
      const end = Math.min(start + batchSize, count);
      console.log("当前批次", start);
      yield tf.tidy(() => {
        const inputs = [];
        const labels = [];
        for (let j = start; j < end; ++j) {
          const { imgPath, dirIndex } = data[j];
          const x = img2x(imgPath);
          inputs.push(x);
          labels.push(dirIndex);
        }
        const xs = tf.concat(inputs);
        const ys = tf.tensor(labels);
        return { xs, ys };
      });
    }
  });

  return { ds, classes };
};

module.exports = getData;
