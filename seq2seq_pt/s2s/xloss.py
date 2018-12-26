import torch


def loss_function(g_outputs, g_targets, generator, crit, eval=False):
    batch_size = g_outputs.size(1)

    g_out_t = g_outputs.view(-1, g_outputs.size(2))
    g_prob_t = generator(g_out_t)

    g_loss = crit(g_prob_t, g_targets.view(-1))
    total_loss = g_loss
    report_loss = total_loss.item()
    return total_loss, report_loss, 0


def generate_copy_loss_function(g_outputs, c_gate_values, c_outputs, g_targets, tgt_mask,
                                extended_src_batch, extended_tgt_batch, extended_vocab_size,
                                generator, crit):
    total_time_step = len(g_outputs)
    batch_size = g_outputs[0].size(0)

    extend_zeros = None
    if extended_vocab_size > 0:
        extend_zeros = torch.zeros((batch_size, extended_vocab_size)).to(g_outputs[0].device)

    all_prob_buf = []
    for i in range(total_time_step):
        c_gate = c_gate_values[i]
        g_digits = g_outputs[i]
        g_prob = generator(g_digits)
        g_prob = (1 - c_gate) * g_prob
        c_prob = c_outputs[i]
        c_prob = c_gate * c_prob

        if extended_vocab_size > 0:
            extend_prob = torch.cat((g_prob, extend_zeros), 1)
            extend_prob = extend_prob.scatter_add(1, extended_src_batch, c_prob)
        else:
            extend_prob = g_prob
        all_prob_buf.append(extend_prob)

    prob = torch.stack(all_prob_buf) + 1e-8
    log_prob = torch.log(prob)
    log_prob = log_prob.view(total_time_step * batch_size, -1)
    targets = extended_tgt_batch.view(-1)
    loss = crit(log_prob, targets)
    loss = loss.view(total_time_step, batch_size)

    loss = (1 - tgt_mask) * loss

    total_loss = torch.sum(loss)
    report_loss = total_loss.item()
    return total_loss, report_loss, 0


def coverage_loss_function(all_coverage, all_attn, tgt_mask):
    coverage = torch.stack(all_coverage)
    attn = torch.stack(all_attn)
    loss = torch.min(coverage, attn)
    loss = torch.sum(loss, 2)
    loss = loss * (1 - tgt_mask)
    loss = torch.sum(loss)

    return loss
