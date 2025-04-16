import os
import shutil
import yaml
import click

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
default_ph_file_path = os.path.join(script_dir, "phoneme_type.yaml")

@click.command(help="OpenUtau voicebank builder.")
@click.option("--acoustic_onnx_folder", required=True, help="Path to the folder containing acoustic ONNX files")
@click.option("--acoustic_config", required=True, help="Path to the config.yaml used for acoustic training")
@click.option("--variance_onnx_folder", required=True, help="Path to the folder containing variance ONNX files")
@click.option("--variance_config", required=True, help="Path to the config.yaml used for variance training")
@click.option("--dictionary_path", required=True, help="Path to the dictionary")
@click.option("--save_path", required=True, help="Voicebank saving directory")
@click.option("--name", default="my_diffsinger_voicebank", help="Character name, default: my_diffsinger_voicebank")
@click.option("--vocoder_onnx_model", required=False, help="Path to the vocoder onnx file itself, not folder")
@click.option("--phoneme_type_file", default=f"{default_ph_file_path}", help="Path to the file specifying phoneme type")

def main(acoustic_onnx_folder, acoustic_config, variance_onnx_folder, variance_config, dictionary_path, save_path, name, vocoder_onnx_model, phoneme_type_file):
    print("\nmaking dsmain directory and necessary files...")
    main_stuff = f"{save_path}/{name}"
    if not os.path.exists(main_stuff):
        os.makedirs(main_stuff)
    if not os.path.exists(f"{main_stuff}/dsmain"):
        os.makedirs(f"{main_stuff}/dsmain/embeds/acoustic")
        os.makedirs(f"{main_stuff}/dsmain/embeds/variance")

    shutil.copy(f"{acoustic_onnx_folder}/acoustic.onnx", f"{main_stuff}/dsmain")
    shutil.copy(f"{acoustic_onnx_folder}/phonemes.txt", f"{main_stuff}/dsmain")
    acoustic_emb_files = [file for file in os.listdir(acoustic_onnx_folder) if file.endswith(".emb")]
    for emb_file in acoustic_emb_files:
        shutil.copy(f"{acoustic_onnx_folder}/{emb_file}", f"{main_stuff}/dsmain/embeds/acoustic")

    shutil.copy(f"{variance_onnx_folder}/linguistic.onnx", f"{main_stuff}/dsmain")
    variance_emb_files = [file for file in os.listdir(variance_onnx_folder) if file.endswith(".emb")]
    for emb_file in variance_emb_files:
        shutil.copy(f"{variance_onnx_folder}/{emb_file}", f"{main_stuff}/dsmain/embeds/variance")

    with open(f"{main_stuff}/character.txt", "w", encoding = "utf-8") as file:
        file.write(f"name={name}\n")
    with open(f"{main_stuff}/character.yaml", "w", encoding = "utf-8") as file: #create initial yaml
        file.write("default_phonemizer: OpenUtau.Core.DiffSinger.DiffSingerPhonemizer\n")
        file.write("singer_type: diffsinger\n")

    acoustic_emb_files = os.listdir(acoustic_onnx_folder)
    acoustic_embeds = []
    acoustic_color_suffix = []
    for file in acoustic_emb_files:
        if file.endswith(".emb"):
            acoustic_emb = os.path.splitext(file)[0]
            acoustic_embeds.append("dsmain/embeds/acoustic/" + acoustic_emb)
            acoustic_color_suffix.append(acoustic_emb)

    subbanks = []

    for i, (acoustic_embed_color, acoustic_embed_suffix) in enumerate(zip(acoustic_color_suffix, acoustic_embeds), start=1):
        color = f"{i:02}: {acoustic_embed_color}"
        suffix = f"{acoustic_embed_suffix}"
        subbanks.append({"color": color, "suffix": suffix})

    if subbanks:
        with open(f"{main_stuff}/character.yaml", "r", encoding = "utf-8") as config:
            character_config = yaml.safe_load(config)
        character_config["subbanks"] = subbanks
        with open(f"{main_stuff}/character.yaml", "w", encoding = "utf-8") as config:
            yaml.dump(character_config, config)

    #image, portrait, and portrait opacity can be manually edited
    with open(f"{main_stuff}/character.yaml", "a", encoding = "utf-8") as file:
        file.write("\n")
        file.write("text_file_encoding: utf-8\n")
        file.write("\n")
        file.write("image:\n")
        file.write("portrait:\n")
        file.write("portrait_opacity: 0.45\n")

    with open(f"{main_stuff}/dsconfig.yaml", "w", encoding = "utf-8") as file:
        file.write("phonemes: dsmain/phonemes.txt\n")
        file.write("acoustic: dsmain/acoustic.onnx\n")
        file.write("vocoder: nsf_hifigan\n")
        file.write("singer_type: diffsinger\n")

    with open(acoustic_config, "r", encoding = "utf-8") as config:
        acoustic_config_data = yaml.safe_load(config)
    use_energy_embed = acoustic_config_data.get("use_energy_embed")
    use_breathiness_embed = acoustic_config_data.get("use_breathiness_embed")
    use_shallow_diffusion = acoustic_config_data.get("use_shallow_diffusion")
    max_depth = acoustic_config_data.get("T_start")
    speakers = acoustic_config_data.get("speakers")
    augmentation_arg = acoustic_config_data.get("augmentation_args")
    pitch_aug = acoustic_config_data.get("use_key_shift_embed")
    time_aug = acoustic_config_data.get("use_speed_embed")
    voicing = acoustic_config_data.get("use_voicing_embed")
    tension = acoustic_config_data.get("use_tension_embed")
    sample_rate = acoustic_config_data.get("audio_sample_rate")
    hop_size = acoustic_config_data.get("hop_size")
    win_size = acoustic_config_data.get("win_size")
    fft_size = acoustic_config_data.get("fft_size")
    num_mel_bins = acoustic_config_data.get("audio_num_mel_bins")
    mel_fmin = acoustic_config_data.get("fmin")
    mel_fmax = acoustic_config_data.get("fmax")
    mel_base = acoustic_config_data.get("mel_base")
    mel_scale = "slaney"


    with open(f"{main_stuff}/dsconfig.yaml", "r", encoding = "utf-8") as config:
        dsconfig_data = yaml.safe_load(config)

    dsconfig_data["use_energy_embed"] = use_energy_embed
    dsconfig_data["use_breathiness_embed"] = use_breathiness_embed
    dsconfig_data["use_variable_depth"] = use_shallow_diffusion
    dsconfig_data["max_depth"] = max_depth
    dsconfig_data["augmentation_args"] = augmentation_arg
    dsconfig_data["use_key_shift_embed"] = pitch_aug
    dsconfig_data["use_speed_embed"] = time_aug
    dsconfig_data["use_voicing_embed"] = voicing
    dsconfig_data["use_tension_embed"] = tension
    dsconfig_data["use_continuous_acceleration"] = True
    dsconfig_data["sample_rate"] = sample_rate
    dsconfig_data["hop_size"] = hop_size
    dsconfig_data["win_size"] = win_size
    dsconfig_data["fft_size"] = fft_size
    dsconfig_data["num_mel_bins"] = num_mel_bins
    dsconfig_data["mel_fmin"] = mel_fmin
    dsconfig_data["mel_fmax"] = mel_fmax
    dsconfig_data["mel_base"] = mel_base
    dsconfig_data["mel_scale"] = mel_scale

    if subbanks:
        dsconfig_data["speakers"] = acoustic_embeds

    with open(f"{main_stuff}/dsconfig.yaml", "w", encoding = "utf-8") as config:
        yaml.dump(dsconfig_data, config)

    variance_emb_files = os.listdir(variance_onnx_folder)
    variance_embeds = []

    for file in variance_emb_files:
        if file.endswith(".emb"):
            variance_emb = os.path.splitext(file)[0]
            variance_embeds.append("../dsmain/embeds/variance/" + variance_emb)
    phoneme_dict_path = f"{acoustic_onnx_folder}/dictionary.txt"
    dsdict = "dsdict.yaml"

    def parse_phonemes(phonemes_str):
        return phonemes_str.split()

    with open(phoneme_type_file, "r", encoding = "utf-8") as ph_type:
        phonemes_data = yaml.safe_load(ph_type)

    vowel_types = set(phonemes_data.get("vowel", []))
    semivowel_types = set(phonemes_data.get("liquid", []))

    entries = []
    vowel_data = []
    semivowel_data = []
    stop_data = []

    with open(dictionary_path, "r", encoding = "utf-8") as f:
        for line in f:
            word, phonemes_str = line.strip().split("\t")
            phonemes = parse_phonemes(phonemes_str)
            entries.append({"grapheme": word, "phonemes": phonemes})

    with open(phoneme_dict_path, "r", encoding = "utf-8") as f:
        for line in f:
            phoneme, _ = line.strip().split("\t")
            phoneme_type = "vowel" if phoneme in vowel_types else ("liquid" if phoneme in semivowel_types else "stop")
            entry = {"symbol": phoneme, "type": phoneme_type}
            if phoneme_type == "vowel":
                vowel_data.append(entry)
            elif phoneme_type == "liquid":
                semivowel_data.append(entry)
            else:
                stop_data.append(entry)

    vowel_data.sort(key=lambda x: x["symbol"])
    semivowel_data.sort(key=lambda x: x["symbol"])
    stop_data.sort(key=lambda x: x["symbol"])

    dsdict_path = os.path.join(main_stuff, dsdict)
    with open(dsdict_path, "w", encoding = "utf-8") as f:
        f.write("entries:\n")
        for entry in entries:
            f.write(f"- grapheme: {entry['grapheme']}\n")
            f.write("  phonemes:\n")
            for phoneme in entry["phonemes"]:
                f.write(f"  - {phoneme}\n")

        f.write("\nsymbols:\n")
        for entry in vowel_data + semivowel_data + stop_data:
            f.write(f"- symbol: {entry['symbol']}\n")
            f.write(f"  type: {entry['type']}\n")

    with open(variance_config, "r", encoding = "utf-8") as config:
        variance_config_data = yaml.safe_load(config)
    sample_rate = variance_config_data.get("audio_sample_rate")
    hop_size = variance_config_data.get("hop_size")
    predict_dur = variance_config_data.get("predict_dur")
    predict_pitch = variance_config_data.get("predict_pitch")
    predict_energy = variance_config_data.get("predict_energy")
    predict_breathiness = variance_config_data.get("predict_breathiness")
    predict_tension = variance_config_data.get("predict_tension")
    predict_voicing = variance_config_data.get("predict_voicing")
    use_melody_encoder = variance_config_data.get("use_melody_encoder")

    dur_onnx_path = variance_onnx_folder + "/dur.onnx"
    if os.path.exists(dur_onnx_path):
        print("making dsdur directory and necessary files...")
        os.makedirs(f"{main_stuff}/dsdur")
        shutil.copy(dur_onnx_path, os.path.join(main_stuff, "dsdur"))
        shutil.copy(dsdict_path, os.path.join(main_stuff, "dsdur"))
        with open(f"{main_stuff}/dsdur/dsconfig.yaml", "w", encoding = "utf-8") as file:
            file.write("phonemes: ../dsmain/phonemes.txt\n")
            file.write("linguistic: ../dsmain/linguistic.onnx\n")
            file.write("dur: dur.onnx\n")
        with open(f"{main_stuff}/dsdur/dsconfig.yaml", "r", encoding = "utf-8") as config:
            dsdur_config = yaml.safe_load(config)
        dsdur_config["use_continuous_acceleration"] = True
        dsdur_config["sample_rate"] = sample_rate
        dsdur_config["hop_size"] = hop_size
        dsdur_config["predict_dur"] = predict_dur
        if subbanks:
            dsdur_config["speakers"] = variance_embeds
        with open(f"{main_stuff}/dsdur/dsconfig.yaml", "w", encoding = "utf-8") as config:
            yaml.dump(dsdur_config, config)
    else:
        print("dur.onnx not found, not making dsdur folder...")

    pitch_onnx_path = variance_onnx_folder + "/pitch.onnx"
    if os.path.exists(pitch_onnx_path):
        print("making dspitch directory and necessary files...")
        os.makedirs(f"{main_stuff}/dspitch")
        shutil.copy(pitch_onnx_path, os.path.join(main_stuff, "dspitch"))
        shutil.copy(dsdict_path, os.path.join(main_stuff, "dspitch"))
        with open(f"{main_stuff}/dspitch/dsconfig.yaml", "w", encoding = "utf-8") as file:
            file.write("phonemes: ../dsmain/phonemes.txt\n")
            file.write("linguistic: ../dsmain/linguistic.onnx\n")
            file.write("pitch: pitch.onnx\n")
            file.write("use_expr: true\n")
        with open(f"{main_stuff}/dspitch/dsconfig.yaml", "r", encoding = "utf-8") as config:
            dspitch_config = yaml.safe_load(config)
        dspitch_config["use_continuous_acceleration"] = True
        dspitch_config["sample_rate"] = sample_rate
        dspitch_config["hop_size"] = hop_size
        dspitch_config["predict_dur"] = predict_pitch
        if subbanks:
            dspitch_config["speakers"] = variance_embeds
        dspitch_config["use_note_rest"] = use_melody_encoder
        with open(f"{main_stuff}/dspitch/dsconfig.yaml", "w", encoding = "utf-8") as config:
            yaml.dump(dspitch_config, config)
    else:
        print("pitch.onnx not found, not making dspitch folder...")

    variance_onnx_path = variance_onnx_folder + "/variance.onnx"
    if os.path.exists(variance_onnx_path):
        print("making dsvariance directory and necessary files...")
        os.makedirs(f"{main_stuff}/dsvariance")
        shutil.copy(variance_onnx_path, os.path.join(main_stuff, "dsvariance"))
        shutil.copy(dsdict_path, os.path.join(main_stuff, "dsvariance"))
        with open(f"{main_stuff}/dsvariance/dsconfig.yaml", "w", encoding = "utf-8") as file:
            file.write("phonemes: ../dsmain/phonemes.txt\n")
            file.write("linguistic: ../dsmain/linguistic.onnx\n")
            file.write("variance: variance.onnx\n")
        with open(f"{main_stuff}/dsvariance/dsconfig.yaml", "r", encoding = "utf-8") as config:
            dsvariance_config = yaml.safe_load(config)
        dsvariance_config["use_continuous_acceleration"] = True
        dsvariance_config["sample_rate"] = sample_rate
        dsvariance_config["hop_size"] = hop_size
        dsvariance_config["predict_dur"] = True
        dsvariance_config["predict_voicing"] = predict_voicing
        dsvariance_config["predict_tension"] = predict_tension
        dsvariance_config["predict_energy"] = predict_energy
        dsvariance_config["predict_breathiness"] = predict_breathiness
        if subbanks:
            dsvariance_config["speakers"] = variance_embeds
        with open(f"{main_stuff}/dsvariance/dsconfig.yaml", "w", encoding = "utf-8") as config:
            yaml.dump(dsvariance_config, config)
    else:
        print("variance.onnx not found, not making dsvariance folder...")

    if vocoder_onnx_model:
        print("making dsvocoder directory and necessary files...")
        os.makedirs(f"{main_stuff}/dsvocoder")
        shutil.copy(vocoder_onnx_model, os.path.join(main_stuff, "dsvocoder"))
        vocoder_onnx_model_file = os.path.basename(vocoder_onnx_model)
        vocoder_onnx_model_folder = os.path.dirname(vocoder_onnx_model)
        vocoder_name = os.path.splitext(vocoder_onnx_model_file)[0]
        with open(f"{main_stuff}/dsvocoder/vocoder.yaml", "w", encoding = "utf-8") as file:
            file.write(f"model: {vocoder_onnx_model_file}\n")
            file.write(f"name: {vocoder_name}\n")
            #should always match with the acoustic model
            file.write(f"sample_rate: {sample_rate}\n")
            file.write(f"hop_size: {hop_size}\n")
            file.write(f"win_size: {win_size}\n")
            file.write(f"fft_size: {fft_size}\n")
            file.write(f"num_mel_bins: {num_mel_bins}\n")
            file.write(f"mel_fmin: {mel_fmin}\n")
            file.write(f"mel_fmax: {mel_fmax}\n")
            file.write(f"mel_base: {mel_base}\n")
            file.write(f"mel_scale: {mel_scale}\n")
        with open(f"{main_stuff}/dsconfig.yaml", "r", encoding = "utf-8") as config:
            dsconfig_data2 = yaml.safe_load(config)
        dsconfig_data2["vocoder"] = "dsvocoder"
        with open(f"{main_stuff}/dsconfig.yaml", "w", encoding = "utf-8") as config:
            yaml.dump(dsconfig_data2, config)
        shutil.copy(os.path.join(vocoder_onnx_model_folder, "config.json"), os.path.join(main_stuff, "dsvocoder"))
    else:
        print("vocoder path not specified, not making dsvocoder folder...")

    print("done!")

    print("You can use your model in OpenUtau. Please edit any config if necessary")

    pass

if __name__ == "__main__":
    main()
