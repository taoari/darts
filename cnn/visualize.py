import sys
import genotypes
from graphviz import Digraph
import re, os


def plot(genotype, filename):
  g = Digraph(
      format='png',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=False)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  if os.path.exists(sys.argv[1]):
    gens = []
    with open(sys.argv[1]) as f:
      for line in f.readlines():
        match = re.match(r'.*(Genotype\(.*\))', line)
        if match is not None:
          gens.append(match.group(1))
    for i, gen in enumerate(gens):
      from genotypes import Genotype
      genotype = eval(gen)
      plot(genotype.normal, "genotype.normal.{}.gv".format(i))
      plot(genotype.reduce, "genotype.reduction.{}.gv".format(i))

  else:

    genotype_name = sys.argv[1]
    try:
      genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
      print("{} is not specified in genotypes.py".format(genotype_name))
      sys.exit(1)

    plot(genotype.normal, "genotye.{}.normal.gv".format(genotype_name))
    plot(genotype.reduce, "genotye.{}.reduction.gv".format(genotype_name))

