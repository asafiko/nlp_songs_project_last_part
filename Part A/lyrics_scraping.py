from PyLyrics import *
import cPickle as pickle


# scraping all avialble songs for all singers in list singers
def scarp_songs(singers):
    for singer in singers:
        try:
            albums = PyLyrics.getAlbums(singer=singer)
            songs_num = 0
            album_num = 0
            songs_list = []
            for a in albums:
                try:
                    for track in a.tracks():
                        try:
                            lyrics = track.getLyrics()
                        except (UnicodeDecodeError, ValueError):
                            continue
                        song = [str(track), lyrics]
                        songs_list.append(song)
                        songs_num += 1
                    album_num += 1
                except UnboundLocalError:
                    print ("error scraping songs for album :" + str(a))
        except UnboundLocalError:
            print ("no songs for:" + str(a))
        print
        print("total songs for " + singer + " " + str(songs_num))
        print("total albums for " + singer + " " + str(album_num))
        with open(singer + '.pkl', 'wb') as output_file:
            pickle.dump(songs_list, output_file, pickle.HIGHEST_PROTOCOL)


# load songs list from a pickle file
def load_song(path):
    with open(path, 'rb') as input_file:
        songs = pickle.load(input_file)
    return songs


def main():
    # scraping all avialble song for the singers in the list sinfers
    singers = ['The Beatles', 'Britney Spears', 'Eminem']
    scarp_songs(singers)

    # checking the num of songs for each singer
    for singer in singers:
        songs_by_pickle = load_song(singer + ".pkl")
        print (singer + " " + str(len(songs_by_pickle)))


if __name__ == "__main__":
    main()
