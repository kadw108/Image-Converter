read -p "Input gif: " gifpath

python3 main.py --brush_width 1 --path $gifpath --gradient scharr --output_path temp_blahblah.gif
