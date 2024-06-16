#coding=utf8
import os
import dac
from audiotools import AudioSignal

# Set CUDA environment variable to use the first GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = '3' ##在linux终端才有用



###以下是所有需要解码的音频
mydict={  "48kHz":{
            "data":["crema_d","esc50","fsd50k","gunshot_triangulation","vox_lingua_top10"],
            # "model":"/project/hyzhang/mywork/src/DAC/zhymodels/48khz_512d8q_7.5kbps/best/dac/weights.pth",
            "model":"/home/huiyu/workspace/Codec/DAC/descript-audio-codec-main/modelres_bak/48khz_512d8q_7.5kbps/best/dac/weights.pth"
            
            },#5

        "44.1kHz":{
            "data":["ESC-50-master"],#1
            # "model":"/project/hyzhang/mywork/src/DAC/zhymodels/44khz_512d8q_7kbps/best/dac/weights.pth",
            "model":"/home/huiyu/workspace/Codec/DAC/descript-audio-codec-main/modelres_bak/44khz_512d8q_7kbps/best/dac/weights.pth"
            },

        "16kHz":{
            "data":["fluent_speech_commands","libri2Mix_test","librispeech","quesst","snips_test_valid_subset","voxceleb1","RAVDESS","vox1_test_wav","LibriSpeech"],#9
            # "model":"/project/hyzhang/mywork/src/DAC/zhymodels/16khz_320d4q_2kbps/best/dac/weights.pth",
            "model":"/home/huiyu/workspace/Codec/DAC/descript-audio-codec-main/modelres_bak/16khz_320d4q_2kbps/best/dac/weights.pth"
            
            },
}


output_dirname = "dac_debug_codecsuperb"
#############################

for tag,subdict in mydict.items():
    data_list=subdict['data']
    model_path=subdict['model']
    # Define the model path and load the DAC model
    model = dac.DAC.load(model_path)
    model.cuda()
    print(f"processing {tag}")
    #######################################
    for name in data_list:
        ##############################
        print(f"processing {tag}---{name}")
        # scpfile= f"/home/hyzhang/project/mywork/src/DAC/data/ref_data/scp/{name}.wav.scp"
        scpfile= f"/home/huiyu/workspace/Codec/Codec-SUPERB-SLT_Challenge/Codec-SUPERB-SLT_Challenge/codec_superb_data/ref_data/scp/{name}.wav.scp"
        wavlist=[]
        with open(scpfile,'r',encoding='utf8')as fr:
            for line in fr:
                line=line.strip()
                wavname,wavpath=line.split()
                wavlist.append(wavpath)
        #######################################
        # Loop through all audio files in the input directory
        for filename in wavlist:
            #####################################
            output_file=filename.replace("ref_data",f"{output_dirname}")
            output_root=os.path.dirname(output_file)
            if not os.path.exists(output_root):
                os.makedirs(output_root,exist_ok=True)
            ##########################
            # Load the audio signal and move it to the GPU
            signal = AudioSignal(filename)
            signal.cuda()
            
            # Compress and decompress the signal
            ###method1
            x = model.compress(signal)
            y = model.decompress(x)
            y = y.detach().cpu()
            ####################
            #method2-容易OOM
            # x = model.preprocess(signal.audio_data, signal.sample_rate)
            # z, codes, latents, _, _ = model.encode(x)
            # y = model.decode(z)
            # y = y.squeeze(0).detach().cpu()
            # y=AudioSignal(y,sample_rate=signal.sample_rate)##############这里的采样率！
            ####################################

            #############################
            # Write the decompressed audio to the output file
            y.write(output_file)
        #############################################
        print(f"processing {tag}--{name} done")
