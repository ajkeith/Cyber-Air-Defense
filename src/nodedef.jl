# tree structure
# each node is a history, which records the id number of the actor who plays
# at the node, the info set it belongs to, the last action taken, and
# the list of children nodes (which each correspond to a next action)
struct Node
    idnode::Int
    idactor::Int
    idinfo::Int
    lastaction::Int
    children::Array{Node}
end

idnode(n::Node) = n.idnode
idinfo(n::Node) = n.idinfo
idactor(n::Node) = n.idactor
children(n::Node) = n.children
lastaction(n::Node) = n.lastaction
nextactions(n::Node) = [m.lastaction for m in children(n)]
function nnodes(n::Node)
    s = 1
    for i = 1:length(n.children)
        s += nnodes(n.children[i])
    end
    s
end
