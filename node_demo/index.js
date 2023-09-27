const tf = require("@tensorflow/tfjs-node-gpu");
const { log } = require("console");
const fs = require("fs");

(async () => {
	const model = await tf.loadLayersModel('file://../output/model-1a/model.json');

	const img2x = (imgPath) => {
		const buffer = fs.readFileSync(imgPath);
		//清除数据
		return tf.tidy(() => {
			//把图片转成tensor
			const imgt = tf.node.decodeImage(new Uint8Array(buffer), 3);
			//调整图片大小
			const imgResize = tf.image.resizeBilinear(imgt, [224, 224]);
			//归一化
			return imgResize.toFloat().sub(255 / 2).div(255 / 2).reshape([1, 224, 224, 3]);
		});
	}
	const pred = tf.tidy(() => {
		const x = img2x('../dataset/train/fire_telephone/01.jpg');
		return model.predict(x);
	});
	const index = pred.argMax(1).dataSync()[0];
	const classes = JSON.parse(fs.readFileSync('../output/classes.json', { encoding: 'utf-8' }))
	console.log(pred.dataSync());
	console.log('result:', classes[index])
})()