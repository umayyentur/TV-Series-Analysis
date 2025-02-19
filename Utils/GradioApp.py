import gradio as gr
import os 
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,'../'))
from ThemeClassifier import ThemeClassifier

def get_themes(theme_list_str, subtitle_path, save_path):
    theme_list = theme_list_str.split(",")
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitle_path, save_path)
    
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]
    
    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ["Theme", "Score"]
    output_chart = gr.BarPlot(
        output_df,
        x="Theme",
        y="Score",  
        title="Series Themes", 
        tooltip=["Theme", "Score"],
        vertical=False,
        width=400,
        height=260
    )
    
    return output_chart

def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classisifers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitle_path = gr.Textbox(label="Subtitle or script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes,
                                                inputs=[theme_list, subtitle_path, save_path],
                                                outputs=[plot])
                    
                    
    iface.launch(share=True)
    
if __name__ == '__main__':
    main()