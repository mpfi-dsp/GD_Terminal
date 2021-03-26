# GD_Terminal
A version of Gold Digger able to run via command line

# How to Use Gold Digger
1. Download GDTerminal folder and place in an accesable location
2. Go to GDTerminal/Prefrences/gdterminal_config.txt and edit "magnification, 43" to "magnification, 0"
3. Install python package requirements in requirements.txt
4. Run gdterminal.py to initialize the program and populate neccesary folders. 
5. Return to GDTerminal/Prefrences/gdterminal_config.txt and set "magnification, 0" to any number. This does NOT affect the code and is only for self-logging.
6. Download Gold Digger network parameters (https://maxplanckflorida-my.sharepoint.com/:u:/g/personal/jerezd_mpfi_org/ERxpVLFwa2REtwLKnD645aIB7-hcdjMUWmQnVyZu1QbTNQ?e=LGV0Vz) and place into pix2pix/checkpoints/golddigger/ 
7. Drag an image into GDTerminal/Input and then run gdterminal.py
8. The output will appear in GDTerminal/Output

# Understanding gdterminal_config
1. "magnification, #" this is only for self-logging purpose. A value of 0 will attempt to initialize GD Terminal.
2. "six nm, yes" If 6 nanometer gold particles (or similar size) appear in the image, set to yes. (default: no)
3. "twelve nm, no" If 12 nanometer gold particles (or similar size) appear in the image, set to yes. (default: no)
4. "eighteen nm, no" If 18 nanometer gold particles (or similar size) appear in the image, set to yes. (default: no)
5-6. Lower and Upper bound for area threshold for 6 nm gold particles, in pixels.
7-8. Lower and Upper bound for area threshold for 6 nm gold particles, in pixels.
9-10. Lower and Upper bound for area threshold for 6 nm gold particles, in pixels.
11. "threshold sensitivity, 4" Adjustment of threshold sensitivity when attempting to floodfill the pix2pix particle assumption. (default: 4)
