
min_turns=5

# if you want to compare two methods, you can use the following command
# python episode_evaluation.py  --share_file1 $SHARE1 --share_file2 $SHARE2 --share_file3 $SHARE3 \
#  --noshare_file1 $NOSHARE1 --noshare_file2 $NOSHARE2 --noshare_file3 $NOSHARE3

# or if you want to compare just one method, you can use the following command
python episode_evaluation_one_file.py --file1 $SHARE1 --file2 $SHARE2 --file3 $SHARE3 --min_turns $min_turns