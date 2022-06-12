// Import template file.

local template = import "template_full.libsonnet";

// Set options.

local params = {
  use_lstm: true,
  longformer_fine_tune: std.extVar("longformer_fine_tune"),
  loss_weights: {          // Loss weights for the modules.
    ner: std.extVar('nw'),
    saliency: std.extVar('lw'),
    n_ary_relation: std.extVar('rw')
  },
  relation_cardinality: std.parseInt(std.extVar('relation_cardinality')),
  exact_match: std.extVar('em')
};

template(params)