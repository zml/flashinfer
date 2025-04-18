def perms(word):
    stack = list(word)
    results = [stack.pop()]
    for i in range(999999):
        if len(stack) == 0:
            break
        c = stack.pop()
        new_results = []
        for w in results:
            for i in range(len(w)+1):
                new_results.append(w[:i] + c + w[i:])
        results = new_results
    return results

def dot_product(x, y):
    dp = 0
    for i in range(len(x)):
        dp += (x[i]*y[i])
    return dp


def _kernel_instances_impl(ctx):
    keys = ctx.attr.values.keys()
    values = ctx.attr.values.values()
    iterators = [0] * len(keys)
    dicts = []
    total = 1
    for v in values:
        total *= len(v)
    for i in range(total):
        v = i
        for j in range(len(iterators)):
            iterators[j] = v % len(values[j])
            v =  v // len(values[j])
        dicts.append({
            keys[i]: values[i][j]
            for i, j in enumerate(iterators)
        })

    files = []
    for i, d in enumerate(dicts):
        for alias, values in ctx.attr.aliases.items():
            if alias in d:
                for target in values:
                    d[target] = d[alias]
        json_data = ctx.actions.declare_file("{}_{}.cu.json".format(ctx.label.name, i))
        ctx.actions.write(
            output = json_data,
            content = json.encode(d),
        )
        cufile_pre_pre_sub = ctx.actions.declare_file("{}_{}.pre_pre_sub.cu".format(ctx.label.name, i))
        ctx.actions.expand_template(
            template = ctx.file.template,
            output = cufile_pre_pre_sub,
            substitutions = ctx.attr.pre_substitutions,
        )
        cufile_pre_sub = ctx.actions.declare_file("{}_{}.pre_sub.cu".format(ctx.label.name, i))
        args = ctx.actions.args()
        args.add_all([
            "-o",
            cufile_pre_sub,
            "--data",
            json_data,
            "--format",
            "json",
            cufile_pre_pre_sub,
        ])
        ctx.actions.run(
            inputs = [json_data, cufile_pre_pre_sub],
            outputs = [cufile_pre_sub],
            executable = ctx.executable._jinja,
            arguments = [args],
        )
        cufile_post_sub = ctx.actions.declare_file("{}_{}.cu".format(ctx.label.name, i))
        ctx.actions.expand_template(
            template = cufile_pre_sub,
            output = cufile_post_sub,
            substitutions = ctx.attr.substitutions,
        )
        files.append(cufile_post_sub)

    return [
        DefaultInfo(
            files = depset(files),
        ),
    ]

kernel_instances = rule(
    implementation = _kernel_instances_impl,
    attrs = {
        "template": attr.label(allow_single_file = True),
        "values": attr.string_list_dict(),
        "pre_substitutions": attr.string_dict(),
        "substitutions": attr.string_dict(),
        "aliases": attr.string_list_dict(),
        "_jinja": attr.label(
            executable = True,
            cfg = "host",
            default = "@@//:jinja_cli",
        ),
    },
)
