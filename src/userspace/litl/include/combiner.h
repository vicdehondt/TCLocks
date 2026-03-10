// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2022 Vishal Gupta, Kumar Kartikeya Dwivedi

/*
 * incoming_rsp_ptr -> rdi
 * outgoing_rsp_ptr -> rsi
 *
 * Assume IRQ is disabled. Will enable when returning.
 */

 /*
 * ============================================================
 * AArch64 port notes
 * ============================================================
 *
 * Register mapping from x86-64 to AArch64:
 *
 *   rsp  (stack pointer)   ->  sp
 *   rax  (return value)    ->  x0
 *   rdi  (1st argument)    ->  x0   (same register on AArch64!)
 *   callq sym              ->  bl sym   (saves return addr in x30, not on stack)
 *   pushq / popq           ->  stp/str [sp,#-16]! / ldp/ldr [sp],#16
 *
 * Callee-saved registers:
 *   x86-64:  rbp rbx r12 r13 r14 r15           (6 registers)
 *   AArch64: x19-x28 x29(fp) x30(lr)           (12 registers = 6 pairs)
 *
 *   We save all 12 AArch64 callee-saved registers.  x86 happened to have
 *   exactly 6, so that was complete coverage.  AArch64 has 12; saving fewer
 *   would leave x25-x28 unsaved across the stack switch if the compiler
 *   chose to use them in the surrounding function.
 *
 * The rdi / x0 problem:
 *   On x86-64, rdi is the 1st argument register and is CALLER-saved.  The
 *   original code relies on rdi surviving the two callq instructions, which
 *   works in practice because those functions take no arguments and the
 *   compiler's register allocator happens to keep the value there.
 *
 *   On AArch64, x0 is BOTH the 1st argument AND the return value.  Every
 *   "bl" call overwrites x0 with the return value.  We solve this by:
 *
 *     switch_to:   mov x19, x0        -- stash first arg in callee-saved x19
 *                                        (x19 is already saved on the stack,
 *                                        so its old value is safe)
 *                  ... bl calls ...
 *                  str x19, [sp,#-16]! -- push stashed first arg onto
 *                                         the new shadow stack
 *
 *     switch_from: ldr x20, [sp], #16  -- pop from shadow stack into x20
 *                                         (callee-saved; bl calls won't
 *                                         clobber it; its original value
 *                                         is already saved on the stack)
 *                  ... bl calls ...
 *                  [now back on original stack]
 *                  mov x0, x20          -- move saved arg to x0 BEFORE the
 *                                         ldp that restores x20 from the
 *                                         original stack would overwrite it
 *                  ldp x19, x20, ...    -- x20 gets its original value back;
 *                                         x0 now holds the shadow-stack arg
 *
 * Stack alignment:
 *   AArch64 requires sp to be 16-byte aligned at all times (hardware
 *   enforced on most cores).  We always push/pop in 16-byte units.
 *   The lone "save first arg" step uses a 16-byte slot (8 bytes live data
 *   + 8 bytes padding) to maintain this invariant.
 *
 * "bl sym" from inline asm:
 *   x86 used "callq %P0" with an "i" (immediate) constraint to embed the
 *   function address.  On AArch64 we write "bl symbol_name" directly in the
 *   asm string.  This is safe for translation-unit-local symbols and for
 *   extern symbols within the +/-128 MB PC-relative branch range, which is
 *   always satisfied within a single shared library.  If your toolchain
 *   places the library beyond that range, replace "bl sym" with:
 *       adrp x9, sym
 *       add  x9, x9, :lo12:sym
 *       blr  x9
 *   and add "x9" to the clobber list.
 *
 * node->rsp offset:
 *   x86 used "%c1(%%rax)" (plain integer offset, no # prefix) to address
 *   a struct field.  On AArch64 we use "str x1, [x0, #%c0]" where %c0
 *   emits the offsetof value as a bare decimal integer, giving the assembler
 *   the "#<N>" immediate it expects.
 */

#define NUMA_AWARE 1

#if defined(__aarch64__) || defined(__arm__)

#define komb_switch_to_shadow_stack()                                          \
    ({                                                                         \
        asm volatile(                                                          \
            /* ── Step 1: push all 12 AArch64 callee-saved registers ──────── \
             * Push outermost pair first; sp ends up pointing at x19/x20.    \
             * After these stores the source registers still hold their        \
             * original values — stp does not clear them.                     \
             */                                                                \
            "stp x29, x30, [sp, #-16]!\n"  /* fp, lr  (outermost) */         \
            "stp x27, x28, [sp, #-16]!\n"                                     \
            "stp x25, x26, [sp, #-16]!\n"                                     \
            "stp x23, x24, [sp, #-16]!\n"                                     \
            "stp x21, x22, [sp, #-16]!\n"                                     \
            "stp x19, x20, [sp, #-16]!\n"  /*        (innermost, sp here) */ \
                                                                               \
            /* ── Step 2: stash first argument before bl clobbers x0 ───────  \
             * x19 is now saved on the stack and free to use as a temp.       \
             */                                                                \
            "mov x19, x0\n"                                                    \
                                                                               \
            /* ── Step 3: node->rsp = sp  (record where our stack lives) ──── \
             * get_komb_mutex_node() returns a pointer to this CPU's          \
             * komb_mutex_node in x0.  We then write sp (via x1, because      \
             * AArch64 str cannot use sp as the value operand with an offset) \
             * into node->rsp.                                                 \
             */                                                                \
            "bl get_komb_mutex_node\n"      /* x0 = &node                 */  \
            "mov x1, sp\n"                  /* x1 = current sp            */  \
            "str x1, [x0, #%c0]\n"          /* node->rsp = sp             */  \
                                                                               \
            /* ── Step 4: switch sp to the shadow stack ─────────────────────  \
             * get_shadow_stack_ptr() returns a pointer to the per-CPU        \
             * shadow-stack pointer variable in x0.                           \
             * We load the actual shadow stack address from *x0 and install   \
             * it as sp.                                                       \
             */                                                                \
            "bl get_shadow_stack_ptr\n"     /* x0 = &shadow_stack_ptr     */  \
            "ldr x1, [x0]\n"               /* x1 = *shadow_stack_ptr     */  \
            "mov sp, x1\n"                  /* sp  =  shadow stack        */  \
                                                                               \
            /* ── Step 5: push stashed first arg onto the shadow stack ──────  \
             * Equivalent of x86 "pushq %%rdi".                               \
             * 16-byte slot: 8 bytes of live data (x19) + 8 bytes pad,        \
             * required to keep sp 16-byte aligned.                           \
             */                                                                \
            "str x19, [sp, #-16]!\n"        /* shadow_stack[--sp] = x0    */  \
            :                                                                  \
            : "i"(offsetof(struct komb_mutex_node, rsp))                      \
              /* Clobbers: x0 (multiple bl return values),                    \
                           x1 (sp temp),                                      \
                           x19 (used to stash first arg),                     \
                           x30 (lr, overwritten by every bl).                 \
                 x20-x28, x29 are unchanged in their registers after stp      \
                 (stp copies but does not clear), and the called functions    \
                 must preserve them (ABI), so they need not be listed.       */ \
            : "memory", "x0", "x1", "x19", "x30");                            \
    })

#define komb_switch_from_shadow_stack()                                        \
    ({                                                                         \
        asm volatile(                                                          \
            /* ── Step 1: pop saved first arg from shadow stack ─────────────  \
             * Reverse of "str x19, [sp, #-16]!" in switch_to.               \
             * We pop into x20 rather than x19 to avoid a later conflict:     \
             * x19's original value (from the ORIGINAL stack) will be         \
             * restored by ldp in step 5.  By using x20 here, we have a      \
             * window to copy it to x0 before that ldp overwrites x20.       \
             * x20 is callee-saved in the ABI, so the bl calls in steps 2-3  \
             * are guaranteed not to clobber it.                              \
             */                                                                \
            "ldr x20, [sp], #16\n"          /* x20 = saved first arg      */  \
                                                                               \
            /* ── Step 2: *shadow_stack_ptr = sp  (save shadow sp) ──────────  \
             * Write the current shadow sp back to the per-CPU variable so    \
             * the shadow stack can be reused in the future.                  \
             */                                                                \
            "bl get_shadow_stack_ptr\n"     /* x0 = &shadow_stack_ptr     */  \
            "mov x1, sp\n"                  /* x1 = current (shadow) sp   */  \
            "str x1, [x0]\n"               /* *shadow_stack_ptr = sp     */  \
                                                                               \
            /* ── Step 3: sp = node->rsp  (restore original stack) ──────────  \
             * Load the saved sp value recorded by switch_to and switch back  \
             * to the original stack.                                          \
             */                                                                \
            "bl get_komb_mutex_node\n"      /* x0 = &node                 */  \
            "ldr x1, [x0, #%c0]\n"         /* x1 = node->rsp             */  \
            "mov sp, x1\n"                  /* sp  = original stack       */  \
                                                                               \
            /* ── Step 4: rescue the saved first arg before it is lost ──────  \
             * We are now on the original stack.  x20 still holds the first   \
             * arg value we popped from the shadow stack.  We must copy it to \
             * x0 RIGHT NOW — the very next ldp will restore x20 from the    \
             * original stack, overwriting our saved value.                   \
             * After this macro returns, x0 holds the value that was in x0   \
             * (≡ rdi) when komb_switch_to_shadow_stack() was called,         \
             * exactly mirroring the x86 "popq %%rdi" behaviour.             \
             */                                                                \
            "mov x0, x20\n"                 /* x0 = rescued first arg     */  \
                                                                               \
            /* ── Step 5: pop all 12 callee-saved registers ──────────────── \
             * Exact reverse of the push sequence in switch_to.              \
             * NOTE: ldp x19, x20 restores x20 from the original stack,     \
             * but we have already saved the shadow-stack value to x0 above. \
             */                                                                \
            "ldp x19, x20, [sp], #16\n"    /*        (innermost, sp here) */ \
            "ldp x21, x22, [sp], #16\n"                                       \
            "ldp x23, x24, [sp], #16\n"                                       \
            "ldp x25, x26, [sp], #16\n"                                       \
            "ldp x27, x28, [sp], #16\n"                                       \
            "ldp x29, x30, [sp], #16\n"    /* fp, lr  (outermost) */         \
            :                                                                  \
            : "i"(offsetof(struct komb_mutex_node, rsp))                      \
            : "memory", "x0", "x1", "x20", "x30");                            \
    })

#else

#define komb_switch_to_shadow_stack()                                          \
    ({                                                                         \
        asm volatile("pushq %%rbp\n"                                           \
                     "pushq %%rbx\n"                                           \
                     "pushq %%r12\n"                                           \
                     "pushq %%r13\n"                                           \
                     "pushq %%r14\n"                                           \
                     "pushq %%r15\n"                                           \
                     :                                                         \
                     :                                                         \
                     : "memory");                                              \
        asm volatile("callq %P0\n"                                             \
                     "movq %%rsp, %c1(%%rax)\n"                                \
                     :                                                         \
                     : "i"(get_komb_mutex_node),                               \
                       "i"(offsetof(struct komb_mutex_node, rsp))              \
                     : "memory");                                              \
        asm volatile("callq %P0\n"                                             \
                     "movq (%%rax), %%rsp\n"                                   \
                     "pushq %%rdi\n"                                           \
                     :                                                         \
                     : "i"(get_shadow_stack_ptr)                               \
                     : "memory");                                              \
    })

#define komb_switch_from_shadow_stack()                                        \
    ({                                                                         \
        asm volatile("popq %%rdi\n"                                            \
                     "callq %P0\n"                                             \
                     "movq %%rsp, (%%rax)\n"                                   \
                     :                                                         \
                     : "i"(get_shadow_stack_ptr)                               \
                     : "memory");                                              \
        asm volatile("callq %P0\n"                                             \
                     "movq %c1(%%rax), %%rsp\n"                                \
                     :                                                         \
                     : "i"(get_komb_mutex_node),                               \
                       "i"(offsetof(struct komb_mutex_node, rsp))              \
                     : "memory");                                              \
        asm volatile("popq %%r15\n"                                            \
                     "popq %%r14\n"                                            \
                     "popq %%r13\n"                                            \
                     "popq %%r12\n"                                            \
                     "popq %%rbx\n"                                            \
                     "popq %%rbp\n"                                            \
                     :                                                         \
                     :                                                         \
                     : "memory");                                              \
    })

#endif

void komb_context_switch(void *incoming_rsp_ptr, void *outgoing_rsp_ptr);
