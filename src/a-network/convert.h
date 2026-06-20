#pragma once
#include "core/types.h"
#include <vector>

extern std::vector<float> g_embed_weight;  // [V, d]
extern std::vector<float> g_proj_weight;   // [d, N]
extern std::vector<float> g_proj_bias;     // [N]

void init_convert_layer();
void token_to_signal(size_t token_id, NetworkView network);
