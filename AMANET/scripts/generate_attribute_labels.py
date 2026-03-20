# Script to generate attribute labels for RSICD dataset
# For each image, check if any of the 40 attributes appear in its captions
# If an attribute appears in any caption, mark it as positive (1), otherwise negative (0)

import json
import argparse
import os

def load_attribute_vocab(attribute_words_file):
    """
    Load attribute vocabulary from file
    """
    with open(attribute_words_file, 'r', encoding='utf-8') as f:
        attribute_words = json.load(f)
    return attribute_words

def load_synonym_mapping(synonym_file):
    """Load synonym mapping from file if provided.

    Expected format:
    {
      "airport": ["airport", "planes", "terminal"],
      ...
    }
    """
    if not synonym_file:
        return None
    if not os.path.exists(synonym_file):
        print(f"Synonym file {synonym_file} not found, ignore and use only canonical words.")
        return None
    with open(synonym_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    print(f"Loaded synonym mapping for {len(mapping)} attributes from {synonym_file}")
    return mapping

def build_attribute_to_synonyms(attribute_words, synonym_mapping):
    """Build attribute -> list of synonyms (lowercased).

    If synonym_mapping is None or does not contain an attribute,
    fall back to using the attribute itself as the only synonym.
    """
    attr_to_synonyms = {}
    for attr in attribute_words:
        syns = None
        if synonym_mapping is not None:
            # Try exact key match first
            syns = synonym_mapping.get(attr)
            if syns is None:
                # Fallback: lowercased key
                syns = synonym_mapping.get(attr.lower())
        if not syns:
            syns = [attr]
        # Normalize to unique lowercased tokens
        clean_syns = sorted({s.lower() for s in syns if isinstance(s, str) and s})
        attr_to_synonyms[attr] = clean_syns
    return attr_to_synonyms

def generate_attribute_labels(opt):
    """
    Generate attribute labels for RSICD dataset
    """
    # Load attribute vocabulary
    print(f"Loading attribute vocabulary from {opt.attribute_words_file}")
    attribute_words = load_attribute_vocab(opt.attribute_words_file)
    print(f"Loaded {len(attribute_words)} attributes")

    # Load synonym mapping (optional)
    synonym_mapping = None
    if hasattr(opt, 'synonym_file') and opt.synonym_file:
        print(f"Loading synonym mapping from {opt.synonym_file}")
        synonym_mapping = load_synonym_mapping(opt.synonym_file)
    else:
        print("No synonym file provided, only canonical attribute words will be used.")

    attribute_to_synonyms = build_attribute_to_synonyms(attribute_words, synonym_mapping)

    # Load dataset
    print(f"Loading dataset from {opt.input_json}")
    with open(opt.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create attribute labels for each image
    print("Generating attribute labels...")
    for image in data['images']:
        # Initialize all attributes as negative (0)
        attribute_labels = [0] * len(attribute_words)
        
        # Check each caption for attribute words / synonyms
        captions = [sentence['raw'].lower() for sentence in image['sentences']]

        # Precompute token set for this image for exact word matching
        token_set = set()
        for sentence in image['sentences']:
            for w in sentence.get('tokens', []):
                token_set.add(w.lower())

        # For each attribute, check if any of its synonyms appears in captions
        for i, attribute in enumerate(attribute_words):
            synonyms = attribute_to_synonyms[attribute]

            # 1) exact token match using tokens
            found = False
            for syn in synonyms:
                if syn in token_set:
                    attribute_labels[i] = 1
                    found = True
                    break

            if found:
                continue
        
        # Add attribute labels to image data
        image['attribute_labels'] = attribute_labels
    
    # Save updated dataset
    print(f"Saving updated dataset to {opt.output_json}")
    with open(opt.output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Also save just the attribute labels for easier use in training
    print(f"Saving attribute labels to {opt.output_labels_file}")
    attribute_labels_data = {}
    for image in data['images']:
        image_id = image['id'] if 'id' in image else image['filename'].split('.')[0]
        attribute_labels_data[image_id] = image['attribute_labels']
    
    with open(opt.output_labels_file, 'w', encoding='utf-8') as f:
        json.dump(attribute_labels_data, f, ensure_ascii=False, indent=2)
    
    print("Attribute label generation completed!")
    
    # Print some statistics
    total_labels = len(data['images']) * len(attribute_words)
    positive_labels = sum(sum(image['attribute_labels']) for image in data['images'])
    negative_labels = total_labels - positive_labels
    
    print(f"\nStatistics:")
    print(f"Total images: {len(data['images'])}")
    print(f"Total attributes: {len(attribute_words)}")
    print(f"Total labels: {total_labels}")
    print(f"Positive labels: {positive_labels} ({positive_labels/total_labels*100:.2f}%)")
    print(f"Negative labels: {negative_labels} ({negative_labels/total_labels*100:.2f}%)")
    
    # Show attribute frequency
    print(f"\nAttribute frequencies:")
    attribute_counts = [0] * len(attribute_words)
    for image in data['images']:
        for i, label in enumerate(image['attribute_labels']):
            if label == 1:
                attribute_counts[i] += 1
    
    # Sort attributes by frequency
    sorted_attributes = sorted(zip(attribute_words, attribute_counts), key=lambda x: x[1], reverse=True)
    for i, (attribute, count) in enumerate(sorted_attributes):
        print(f"{i+1:2d}. {attribute:<15} ({count} images)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate attribute labels for RSICD dataset')
    parser.add_argument('--input_json', type=str, default='data/rsicd.json',
                        help='path to the json file containing the dataset')
    parser.add_argument('--attribute_words_file', type=str, default='data/attribute_words.json',
                        help='path to the attribute words list')
    parser.add_argument('--synonym_file', type=str, default='',
                        help='path to the attribute synonym mapping file (optional)')
    parser.add_argument('--output_json', type=str, default='data/rsicd_with_attributes.json',
                        help='path to save the dataset with attribute labels')
    parser.add_argument('--output_labels_file', type=str, default='data/attribute_labels.json',
                        help='path to save just the attribute labels')
    
    opt = parser.parse_args()
    generate_attribute_labels(opt)