from thuban.catalog import filter_for_visible_stars, load_raw_hipparcos_catalog

if __name__ == "__main__":
    raw_hipparcos = load_raw_hipparcos_catalog()
    reduced_hipparcos = filter_for_visible_stars(raw_hipparcos, dimmest_magnitude=14)
    reduced_hipparcos.to_csv("../thuban/data/reduced_hip.csv", index=False)
