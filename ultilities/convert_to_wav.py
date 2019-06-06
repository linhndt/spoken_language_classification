from pydub import AudioSegment
import os


def convert_mp3_to_wav(data_folder):
    """
    Convert from 'mp3' format file into 'wav' format file

    """

    file_list = os.listdir(data_folder)

    for file in file_list:
        file_name = file[:-4]

        sound = AudioSegment.from_mp3(data_folder + '/' + file)

        sound.export(os.path.join(data_folder, file_name + '.wav'), format='wav')

    for file in os.listdir(data_folder):
        if file.endswith('mp3'):
            os.unlink(os.path.join(data_folder, file))




