import os
import shutil
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description="OpenUtau Voicebank Builder")
    parser.add_argument("--acoustic_onnx_folder", required=True, help="Path to the folder containing acoustic ONNX files (required)")
    parser.add_argument("--acoustic_config", required=True, help="Path to the config.yaml used for acoustic training (required)")
    parser.add_argument("--variance_onnx_folder", required=True, help="Path to the folder containing variance ONNX files (required)")
    parser.add_argument("--variance_config", required=True, help="Path to the config.yaml used for variance training (required)")
    parser.add_argument("--dictionary_path", required=True, help="Path to the dictionary (required)")
    parser.add_argument("--save_path", required=True, help="Voicebank saving directory (required)")
    parser.add_argument("--name", default="my_diffsinger_voicebank", help="Character name, default: my_diffsinger_voicebank (optional)")
    parser.add_argument("--vocoder_onnx_model", required=False, help="Path to the vocoder onnx file itself, not folder (optional)")
    args = parser.parse_args()

    acoustic_onnx_folder = args.acoustic_onnx_folder
    acoustic_config = args.acoustic_config
    variance_onnx_folder = args.variance_onnx_folder
    variance_config = args.variance_config
    dictionary_path = args.dictionary_path
    save_path = args.save_path
    name = args.name
    vocoder_onnx_model = args.vocoder_onnx_model

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

    with open(f"{main_stuff}/character.txt", "w") as file:
        file.write(f"name={name}\n")
    #image, portrait, and portrait opacity can be manually edited
    with open(f"{main_stuff}/character.yaml", "w") as file:
        file.write("text_file_encoding: utf-8\n")
        file.write("#image::\n")
        file.write("#portrait:\n")
        file.write("#portrait_opacity: 0.45\n")
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
        with open(f"{main_stuff}/character.yaml", "r") as config:
            character_config = yaml.safe_load(config)
        character_config["subbanks"] = subbanks
        with open(f"{main_stuff}/character.yaml", "w") as config:
            yaml.dump(character_config, config)

    with open(f"{main_stuff}/dsconfig.yaml", "w") as file:
        file.write("phonemes: dsmain/phonemes.txt\n")
        file.write("acoustic: dsmain/acoustic.onnx\n")
        file.write("vocoder: nsf_hifigan\n")
        file.write("singer_type: diffsinger\n")

    with open(acoustic_config, "r") as config:
        acoustic_config_data = yaml.safe_load(config)

    use_energy_embed = acoustic_config_data.get("use_energy_embed")
    use_breathiness_embed = acoustic_config_data.get("use_breathiness_embed")
    use_shallow_diffusion = acoustic_config_data.get("use_shallow_diffusion")
    max_depth = acoustic_config_data.get("K_step")
    speakers = acoustic_config_data.get("speakers")
    augmentation_arg = acoustic_config_data.get("augmentation_args")
    pitch_aug = acoustic_config_data.get("use_key_shift_embed")
    time_aug = acoustic_config_data.get("use_speed_embed")

    with open(f"{main_stuff}/dsconfig.yaml", "r") as config:
        dsconfig_data = yaml.safe_load(config)

    dsconfig_data["use_energy_embed"] = use_energy_embed
    dsconfig_data["use_breathiness_embed"] = use_breathiness_embed
    dsconfig_data["use_shallow_diffusion"] = use_shallow_diffusion
    dsconfig_data["max_depth"] = max_depth
    dsconfig_data["augmentation_args"] = augmentation_arg
    dsconfig_data["use_key_shift_embed"] = pitch_aug
    dsconfig_data["use_speed_embed"] = time_aug

    if subbanks:
        dsconfig_data["speakers"] = acoustic_embeds

    with open(f"{main_stuff}/dsconfig.yaml", "w") as config:
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

    entries = []
    vowel_types = {"a", "i", "u", "e", "o", "N", "M", "NG", "cl", "vf", "AP", "SP"} #can be edited manually(?)
    vowel_data = []
    stop_data = []

    with open(dictionary_path, "r", encoding="utf-8") as f:
        for line in f:
            word, phonemes_str = line.strip().split("\t")
            phonemes = parse_phonemes(phonemes_str)
            if len(phonemes) == 1:
                entries.append({"grapheme": word, "phonemes": phonemes})
            else:
                entries.append({"grapheme": word, "phonemes": phonemes})

    with open(phoneme_dict_path, "r", encoding="utf-8") as f:
        for line in f:
            phoneme, _ = line.strip().split("\t")
            phoneme_type = "vowel" if phoneme[0] in vowel_types else "stop"
            entry = {"symbol": phoneme, "type": phoneme_type}
            if phoneme_type == "vowel":
                vowel_data.append(entry)
            else:
                stop_data.append(entry)

    vowel_data.sort(key=lambda x: x["symbol"])
    stop_data.sort(key=lambda x: x["symbol"])

    dsdict_path = os.path.join(main_stuff, dsdict)
    with open(dsdict_path, "w") as f:
        f.write("entries:\n")
        for entry in entries:
            f.write(f"- grapheme: {entry['grapheme']}\n")
            f.write("  phonemes:\n")
            for phoneme in entry["phonemes"]:
                f.write(f"  - {phoneme}\n")

        f.write("\nsymbols:\n")
        for entry in vowel_data + stop_data:
            f.write(f"- symbol: {entry['symbol']}\n")
            f.write(f"  type: {entry['type']}\n")

    with open(variance_config, "r") as config:
        mfking_config = yaml.safe_load(config)
    sample_rate = mfking_config.get("audio_sample_rate")
    hop_size = mfking_config.get("hop_size")
    predict_dur = mfking_config.get("predict_dur")
    predict_pitch = mfking_config.get("predict_pitch")
    use_melody_encoder = mfking_config.get("use_melody_encoder")

    dur_onnx_path = variance_onnx_folder + "/dur.onnx"
    if os.path.exists(dur_onnx_path):
        print("making dsdur directory and necessary files...")
        os.makedirs(f"{main_stuff}/dsdur")
        shutil.copy(dur_onnx_path, os.path.join(main_stuff, "dsdur"))
        shutil.copy(dsdict_path, os.path.join(main_stuff, "dsdur"))
        with open(f"{main_stuff}/dsdur/dsconfig.yaml", "w") as file:
            file.write("phonemes: ../dsmain/phonemes.txt\n")
            file.write("linguistic: ../dsmain/linguistic.onnx\n")
            file.write("dur: dur.onnx\n")
        with open(f"{main_stuff}/dsdur/dsconfig.yaml", "r") as config:
            dsdur_config = yaml.safe_load(config)
        if subbanks:
            dsdur_config["speakers"] = variance_embeds
        with open(f"{main_stuff}/dsdur/dsconfig.yaml", "w") as config:
            yaml.dump(dsdur_config, config)
    else:
        print("dur.onnx not found, not making dsdur folder...")

    pitch_onnx_path = variance_onnx_folder + "/pitch.onnx"
    if os.path.exists(pitch_onnx_path):
        print("making dspitch directory and necessary files...")
        os.makedirs(f"{main_stuff}/dspitch")
        shutil.copy(pitch_onnx_path, os.path.join(main_stuff, "dspitch"))
        shutil.copy(dsdict_path, os.path.join(main_stuff, "dspitch"))
        with open(f"{main_stuff}/dspitch/dsconfig.yaml", "w") as file:
            file.write("phonemes: ../dsmain/phonemes.txt\n")
            file.write("linguistic: ../dsmain/linguistic.onnx\n")
            file.write("pitch: pitch.onnx\n")
            file.write("use_expr: true\n")
        with open(f"{main_stuff}/dspitch/dsconfig.yaml", "r") as config:
            dspitch_config = yaml.safe_load(config)
        if subbanks:
            dspitch_config["speakers"] = variance_embeds
        dspitch_config["use_note_rest"] = use_melody_encoder
        with open(f"{main_stuff}/dspitch/dsconfig.yaml", "w") as config:
            yaml.dump(dspitch_config, config)
    else:
        print("pitch.onnx not found, not making dspitch folder...")

    variance_onnx_path = variance_onnx_folder + "/variance.onnx"
    if os.path.exists(variance_onnx_path):
        print("making dsvariance directory and necessary files...")
        os.makedirs(f"{main_stuff}/dsvariance")
        shutil.copy(variance_onnx_path, os.path.join(main_stuff, "dsvariance"))
        shutil.copy(dsdict_path, os.path.join(main_stuff, "dsvariance"))
        with open(f"{main_stuff}/dsvariance/dsconfig.yaml", "w") as file:
            file.write("phonemes: ../dsmain/phonemes.txt\n")
            file.write("linguistic: ../dsmain/linguistic.onnx\n")
            file.write("variance: variance.onnx\n")
        with open(f"{main_stuff}/dsvariance/dsconfig.yaml", "r") as config:
            dsvariance_config = yaml.safe_load(config)
        if subbanks:
            dsvariance_config["speakers"] = variance_embeds
        with open(f"{main_stuff}/dsvariance/dsconfig.yaml", "w") as config:
            yaml.dump(dsvariance_config, config)
    else:
        print("variance.onnx not found, not making dsvariance folder...")

    if vocoder_onnx_model:
        print("making dsvocoder directory and necessary files...")
        os.makedirs(f"{main_stuff}/dsvocoder")
        shutil.copy(vocoder_onnx_model, os.path.join(main_stuff, "dsvocoder"))
        vocoder_onnx_model_file = os.path.basename(vocoder_onnx_model)
        with open(f"{main_stuff}/dsvocoder/dsconfig.yaml", "w") as file:
            file.write(f"model: {vocoder_onnx_model_file}\n")

    print("done!")

    print("You can use your model in OpenUtau. Please edit any config if necessary")

if __name__ == "__main__":
    main()