SHARE1="../results/update_1008_3_session.json"
SHARE2="../results/update_1008_4_session.json"
SHARE3="../results/update_1008_5_session.json"
NOSHARE1="../results/noshare_1008_3_session.json"
NOSHARE2="../results/noshare_1008_4_session.json"
NOSHARE3="../results/noshare_1008_5_session.json"
python episode_evaluation.py  --share_file1 $SHARE1 --share_file2 $SHARE2 --share_file3 $SHARE3 \
 --noshare_file1 $NOSHARE1 --noshare_file2 $NOSHARE2 --noshare_file3 $NOSHARE3