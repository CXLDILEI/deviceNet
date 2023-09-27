const tf = require('@tensorflow/tfjs-node-gpu');
const getData = require('./data');
const TRAIN_PATH = '../dataset/train';
const VALIDATION_PATH = '../dataset/validation'
const OUT_PUT = 'output';
const MOBILENET_URL = 'http://storage.codelab.club/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';
// const MOBILENET_URL = 'http://8.134.131.255:6066/files/mobilenet.json';

(async () => {
	const { ds, classes } = await getData(TRAIN_PATH, OUT_PUT);
	const { ds: validationDs } = await getData(VALIDATION_PATH, OUT_PUT, false)
	// 引入别人训练好的模型
	const mobilenet = await tf.loadLayersModel(MOBILENET_URL);
	// //查看模型结构
	mobilenet.summary();

	const model = tf.sequential();
	//截断模型，复用了86个层
	for (let i = 0; i < 86; ++i) {
		const layer = mobilenet.layers[i];
		layer.trainable = false;
		model.add(layer);
	}
	//降维，摊平数据
	model.add(tf.layers.flatten());
	//设置全连接层
	model.add(tf.layers.dense({
		units: 10,
		activation: 'relu'//设置激活函数，用于处理非线性问题
	}));

	model.add(tf.layers.dense({
		units: classes.length,
		activation: 'softmax'//用于多分类问题
	}));
	//设置损失函数，优化器
	model.compile({
		loss: 'sparseCategoricalCrossentropy',
		optimizer: tf.train.adam(),
		metrics:['acc']
	});

	//训练模型
	await model.fitDataset(ds, { epochs: 10, validationData: validationDs });
	//保存模型
	model.save('file://../output/model-1a');
})();
