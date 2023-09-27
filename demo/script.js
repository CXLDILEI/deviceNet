import * as tf from '@tensorflow/tfjs';
import { img2x, file2img } from './utils';

const MODEL_PATH = 'http://127.0.0.1:8080/t7';
const CLASSES = ["fire_hydrant_pump_start_button","fire_telephone","heat_fire_detector","kobe","smoke_fire_detector"];


window.onload = async () => {
	const model = await tf.loadLayersModel('http://8.134.131.255:6066/files/model-1a/model.json');

	window.predict = async (file) => {
		const img = await file2img(file);
		document.body.appendChild(img);
		const pred = tf.tidy(() => {
			const x = img2x(img);
			return model.predict(x);
		});

		const index = pred.argMax(1).dataSync()[0];
		console.log(pred)
		console.log(pred.argMax(1).dataSync());

		let predictStr = "";
		if (typeof CLASSES[index] == 'undefined') {
			predictStr = BRAND_CLASSES[index];
		} else {
			predictStr = CLASSES[index];
		}

		setTimeout(() => {
			alert(`预测结果：${predictStr}`);
		}, 0);
	};
};
