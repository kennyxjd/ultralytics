from ultralytics import YOLO
import gradio as gr

class MyClassify:
    def __init__(self, type='train'):
        if type == 'train':
            self.DATA = r"C:\Projects\mmpretrain\datasets\ocr_data"
            self.EPOCHS = 50
            self.BATCH = 8
            self.WORKERS = 1
            self.model = YOLO(task='classify', model=r"C:\Projects\ultralytics\ultralytics\cfg\models\v8\ocr-cls.yaml")
            self.train()
        elif type == 'test':
            self.model = YOLO(r"C:\Projects\ultralytics\runs\classify\train16\weights\best.pt")

        elif type == 'demo':
            self.inferencer = YOLO(r"C:\Projects\ultralytics\runs\classify\train16\weights\best.pt")


    def train(self):
        self.model.train(data=self.DATA, epochs=self.EPOCHS, batch=self.BATCH, workers=self.WORKERS)

    def test(self, img_path):
        results = self.model(img_path)
        print(results[0].probs.data.argmax().item())

    def demo(self):
        self.create_ui()

    def create_ui(self):
        with gr.Column():
            in_image = gr.Image(
                label='Input',
                source='upload',
                elem_classes='input_image',
                interactive=True,
                tool='editor',
            )
        with gr.Column():
            out_cls = gr.Label(
                label='Result',
                num_top_classes=5,
                elem_classes='cls_result',
            )
            run_button = gr.Button(
                'Run',
                elem_classes='run_button',
            )
            run_button.click(
                self.inference,
                inputs=[in_image],
                outputs=out_cls,
            )

    def inference(self, image):
        image = image[:, :, ::-1]
        inferencer = self.inferencer

        result = inferencer(image)[0].probs.data.to('cpu').tolist()

        return dict(zip(('Other', 'MA'), result))

if __name__ == '__main__':
    myclassify = MyClassify(type='demo')
    title = '携带材料检测 Demo'
    with gr.Blocks(analytics_enabled=False, title=title) as demo:
        gr.Markdown(f'# {title}')
        with gr.Tabs():
            with gr.TabItem('Image Classification'):
                myclassify.demo()

    demo.launch(share=False, server_name='192.168.253.130')
    # myclassify.test(r"C:\Projects\mmpretrain\datasets\ocr_data\test\20230719-150301.jpg")

