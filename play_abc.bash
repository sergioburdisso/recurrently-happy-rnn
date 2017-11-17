abc2midi $1 -o _tmp_song.mid
timidity _tmp_song.mid
rm $1 _tmp_song.mid