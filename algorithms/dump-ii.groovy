import org.lenskit.knn.item.model.ItemItemModel

def f = new File(args[0])

f.withPrintWriter { pw -> 
    pw.println("item,neighbor,similarity")

    model = recommender.get(ItemItemModel)
    for (long item: model.itemUniverse) {
        for (ne in model.getNeighbors(item).entrySet()) {
            def nbr = ne.key
            def sim = ne.value
            pw.println("$item,$nbr,$sim")
        }
    }
}
