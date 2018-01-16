from read_data import read_species_list

species = read_species_list()


def get_number_of_species():
    return len(species)


if __name__ == '__main__':
    print(species.head())

