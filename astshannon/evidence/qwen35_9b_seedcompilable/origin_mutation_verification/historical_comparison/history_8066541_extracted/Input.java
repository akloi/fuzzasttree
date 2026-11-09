import java.util.Collection;

public class CollectionUtils {

    public static boolean nullsOnly(Collection<?> collection) {

        return collection.stream().
                reduce(true,
                        (collector, value) -> collector && (value = null),
                        (collector1, collector2) -> collector1 && collector2);

    }
}
