# Dog Breed Recommender (Streamlit)

## Repo structure
- app.py                # Streamlit app (main)
- requirements.txt      # Python dependencies
- data/
  - breed_traits.csv
  - trait_descriptions.csv

## How the app works
- Asks the user to rate traits one-by-one (Back / Next / Skip).
- Calculates breed scores = sum(breed_trait_value * user_importance).
- Recommends top 3 breeds and shows:
  - Image (fetched from Dog-Breeds-Dataset on GitHub)
  - Short slideshow video auto-generated from breed images (MoviePy)
  - Auto-generated short description of strong traits
- If GitHub images cannot be fetched in the environment, the app shows a clickable link to the breed folder on GitHub as fallback.

## Live Demo
- You can explore the app live on Streamlit Cloud: Click here to open the app
- The app is fully interactive, and all features are available online.


