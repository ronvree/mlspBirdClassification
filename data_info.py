from read_data import read_species_list

# Static information about the bird species
species = read_species_list()


def get_number_of_species():
    """
    :return: The number of different species present in the data set
    """
    return len(species)


if __name__ == '__main__':
    print(species.head())

